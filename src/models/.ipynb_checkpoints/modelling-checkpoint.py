# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Basics
import pandas as pd
import numpy as np
import os
import random
import re

# Time
import time

# ML Modelling
## Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

## Models
from sklearn import linear_model, feature_selection
from sklearn.ensemble import RandomForestRegressor as RFR
from xgboost import XGBRegressor
import xgboost as xgb_pack
from sklearn.svm import SVR

# ANN
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, Dropout

## Evaluation
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import statsmodels.api as sm

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# pandas settings
pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 1500)

# plot resolution
plt.rcParams["figure.dpi"] = 200       # display resolution
plt.rcParams["savefig.dpi"] = 300 


# %%
# %config InlineBackend.figure_format = 'png'

# %%
class model_data:

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def restrict(self,
                 area_max=1000000000,
                 area_min=0,
                 eui_max=500,
                 eui_min=0,
                 climates=None,
                 classes=None,
                 facilities=None, # this should be used only in the case when exact match facilities need to be filtered
                 facility=None, # this is used when a general type wants to be filtered i.e. office
                 year_built=1900,
                 year_exc=[2021]):
        
        # Function to restrict model data based on criteria defined within the function
        
        # If climates or facilities are not provided, use all unique values from the dataframe
        if climates is None:
            climates = list(self.dataframe['climate'].unique())
        if classes is None:
            classes = list(self.dataframe['building_class'].unique())
        if facilities is None:
            facilities = list(self.dataframe['facility_type'].unique())
        
        # Filter based on the provided criteria
        self.dataframe = self.dataframe[
            (self.dataframe['floor_area_m2'] > area_min) & 
            (self.dataframe['floor_area_m2'] < area_max) & 
            (self.dataframe['site_eui_kwh_m2'] > eui_min) & 
            (self.dataframe['site_eui_kwh_m2'] < eui_max) & 
            (self.dataframe['climate'].isin(climates)) &
            (self.dataframe['building_class'].isin(classes)) & 
            (self.dataframe['facility_type'].isin(facilities)) & 
            (self.dataframe['year_built'] > year_built) &
            ~(self.dataframe['year'].isin(year_exc))
        ]
        if facility is not None:
            self.dataframe = self.dataframe[self.dataframe['facility_type'].str.contains(facility)]
        else:
            pass
        return self

    def dummify(self, columns):
        # Create dummy variables for categorical columns
        self.dataframe = pd.get_dummies(self.dataframe, columns=columns, drop_first=True, dtype=int)
        return self

    def get_data(self):
        # Method to retrieve the processed dataframe
        return self.dataframe

class ml_model:

    def __init__(self, model):
        self.model = model

    # def load_data(self, X_cols, y_col, train_data, test_data):
    #     self.X_train = train_data[X_cols].astype('float32')
    #     self.y_train = train_data[y_col].astype('float32')
    #     self.X_test = test_data[X_cols].astype('float32')
    #     self.y_test = test_data[y_col].astype('float32')

    def load_data(self, col_idx, X_all, Y_all, train_idx, test_idx):
        self.X_train = X_all[train_idx][:, col_idx]
        self.y_train = Y_all[train_idx]
        self.X_test  = X_all[test_idx][:, col_idx]
        self.y_test  = Y_all[test_idx]

    def transformation(self, transform_func=None, inverse_func=None):
        if transform_func:
            self.y_train_transformed = transform_func(self.y_train)
        else:
            self.y_train_transformed = self.y_train

        self.inverse_func = inverse_func

    def train(self, is_ann=False, **kwargs):
        start_time = time.time()
        
        if is_ann:
            self.history = self.model.fit(self.X_train, self.y_train_transformed, **kwargs)
        else:
            self.model.fit(self.X_train, self.y_train_transformed, **kwargs)
            
        elapsed_time = time.time() - start_time
        print(f"Elapsed time to compute the model: {elapsed_time:.2f} seconds")

    def predict_evaluate(self, verbose=True, return_data=False):
        self.y_pred_transformed = self.model.predict(self.X_test)
        
        if self.inverse_func:
            self.y_pred = self.inverse_func(self.y_pred_transformed)
        else:
            self.y_pred = self.y_pred_transformed

        mae = mean_absolute_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        r2 = r2_score(self.y_test, self.y_pred)
        mape = mean_absolute_percentage_error(self.y_test, self.y_pred)

        if verbose:
            print(f"The MAE score is: {mae:,.2f}\n")
            print(f"The RMSE score is: {rmse:,.2f}\n")
            print(f"The R2 score is: {r2:,.2f}")
            print(f"The MAPE score is: {mape:,.2f}\n")

        if return_data:
            return mae, rmse, r2, mape

    def plot_predictions(self, y_col, model_name, save):
        fig, ax= plt.subplots()
        ax.scatter(self.y_pred,self.y_test, s=5) ,

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        plt.set_cmap('Set1')
        ax.plot(lims, lims, c="red")
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"{model_name} Predicted vs Actual \n {y_col}")
        ax.set_xlabel(f"Predicted {y_col}")
        ax.set_ylabel(f"Actual {y_col}")
        ax.legend()
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(path,'outputs','model-figures',f'figure-{model_name}-predictions.png')) 
            plt.savefig(os.path.join(path,'outputs','model-figures',f'figure-{model_name}-predictions.eps')) 
        plt.show()

    def plot_residuals(self, y_col, model_name, save, title="Residual Plot"):
        """
        Plot residuals and their distribution.
        """
        residuals = self.y_test - self.y_pred
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residuals vs Predicted
        sns.scatterplot(x=self.y_pred, y=residuals, ax=axs[0])
        axs[0].axhline(0, color='red', linestyle='--')
        axs[0].set_xlabel("Predicted values")
        axs[0].set_ylabel("Residuals (y_true - y_pred)")
        axs[0].set_title(f"{model_name} \n Residuals vs Predicted")
        
        # Residuals distribution
        sns.histplot(residuals, bins=30, kde=True, ax=axs[1])
        axs[1].set_xlabel("Residuals")
        axs[1].set_title("Residuals Distribution")
        
        plt.suptitle(title)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(path,'outputs','model-figures',f'figure-{model_name}-residuals.png')) 
            plt.savefig(os.path.join(path,'outputs','model-figures',f'figure-{model_name}-residuals.eps')) 
        plt.show()

    def access_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_train_transformed, self.y_test, self.y_pred, self.y_pred_transformed


# Randomise train / test split based on building ID

def randomise_X_y(data, split=0.8, random_seed=3):
    """
    This functions takes building ID and splits data into train / test based on it. Since
    there are more than one row per building, this is to ensure no leakage to test data.
    """
    
    random.seed(random_seed) # set random seed
    
    # Get building ids
    buildings = pd.DataFrame(data.id.unique())
    
    # Generate random idex numbers 
    random_x = random.sample(range(-1, len(buildings)-1), int(len(buildings) * split))
    random_x.sort()
    random_y = [item for item in range(random_x[0], random_x[-1]+1) if item not in random_x]

    #Train split
    buildings_x = buildings.iloc[random_x]

    buildings_x_list = buildings_x[0].values.tolist()

    train = data[data['id'].isin(buildings_x_list)]

    # Test split
    buildings_y = buildings.iloc[random_y]

    buildings_y_list = buildings_y[0].values.tolist()

    test = data[data['id'].isin(buildings_y_list)]

    return train, test

def randomise_X_y_idx(data, split=0.8, random_seed=3, val_split=False, id_column='id'):
    """
    Split data into train/test (and optionally validation) based on unique building IDs to avoid leakage.
    
    Parameters
    ----------
    data : pandas DataFrame
        Must contain column 'id'.
    split : float
        Fraction of buildings to assign to training (remaining go to test).
    random_seed : int
        Random seed for reproducibility.
    val_split : False or float
        If False, no validation split is returned.
        If float between 0 and 1, fraction of training buildings to assign to validation.
    
    Returns
    -------
    If val_split=False:
        train_idx, test_idx : np.ndarray
    If val_split=float:
        train_idx, val_idx, test_idx : np.ndarray
    """
    
    random.seed(random_seed)
    
    # Unique building IDs
    buildings = data[id_column].unique()
    n_train_buildings = int(len(buildings) * split)
    
    # Randomly select train building IDs
    train_buildings = random.sample(list(buildings), n_train_buildings)
    
    # Remaining buildings go to test
    test_buildings = [b for b in buildings if b not in train_buildings]
    
    # Optional validation split
    if val_split:
        n_val_buildings = int(len(buildings) * val_split)
        val_buildings = random.sample(train_buildings, n_val_buildings)
        # Remaining train buildings after removing validation
        train_buildings = [b for b in train_buildings if b not in val_buildings]
    else:
        val_buildings = []

    # Get row indices
    train_idx = data.index[data[id_column].isin(train_buildings)].to_numpy()
    test_idx  = data.index[data[id_column].isin(test_buildings)].to_numpy()
    if val_split:
        val_idx = data.index[data[id_column].isin(val_buildings)].to_numpy()
        return train_idx, val_idx, test_idx
    else:
        return train_idx, test_idx

# Constants
sq_ft_to_sq_m = 0.09290304
kbtu_to_kwh = 0.29307107017
ft_to_m = 0.3048
kbtu_ft2_to_kwh_m2 = 3.15459

# %%
# Data pre-engineering
path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())))

# Load data
chunks = pd.read_csv(os.path.join(path, 'src', 'data-exploration', 'bpd-weather-enriched.csv'), chunksize=100_000)
bpd_raw = pd.concat(chunks, ignore_index=True)

# Load embeddings
embeddings_path = os.path.join(path, 'data', 'embeddings','zcta_embeddings.csv')
embeddings = pd.read_csv(embeddings_path)

# Load data descriptions
data_descriptions = pd.read_csv(os.path.join(path, 'src', 'data-exploration', 'noaa-data-descriptions.csv'))

# Process embeddings zipcodes
embeddings[['type','zip_code']] = embeddings['place'].str.split("/", expand=True)

# Drop missing zipcodes
bpd_raw = bpd_raw[bpd_raw['zip_code']!='Unknown']

# Set as integers before merge
embeddings['zip_code'] = embeddings['zip_code'].astype(int)
bpd_raw['zip_code'] = bpd_raw['zip_code'].astype(int)

# merge full dataframe
bpd_full = bpd_raw.merge(
    embeddings,
    on='zip_code',
    how='left'
)

# Remove NaNs
bpd_full = bpd_full.replace(['No Value', 'Unknown'], np.nan)

# Filter missing values for important columns
bpd_full = bpd_full[bpd_full.site_eui.notna()]

# Filter columns with missing values
threshold = 0.55 
bpd_full = bpd_full.loc[:, bpd_full.isna().mean() < threshold]

#Transform values into numerical ones
bpd_full['year_built'] = pd.to_numeric(bpd_full["year_built"], errors="coerce").astype("Int64")
bpd_full['energy_star_rating'] = bpd_full.energy_star_rating.astype('float')
bpd_full['site_eui'] = bpd_full.site_eui.astype('float')
bpd_full['fuel_eui'] = bpd_full.fuel_eui.astype('float')
bpd_full['electric_eui'] = bpd_full.electric_eui.astype('float')
bpd_full['ghg_emissions_int'] = bpd_full.ghg_emissions_int.astype('float')

# Replace climate codes
bpd_full['climate_trimmed'] = bpd_full['climate'].str.replace(r"\s*\(.*\)", "", regex=True)
bpd_full['climate_code'] = bpd_full['climate'].str.extract(r"(\d+[A-Z])")[0]


# Translate units & create some useful variables
bpd_full['floor_area_m2'] = bpd_full['floor_area'] * sq_ft_to_sq_m
bpd_full['site_energy_kBTU'] = bpd_full['floor_area'] * bpd_full['site_eui']
bpd_full['site_energy_kwh'] = bpd_full['site_energy_kBTU'] * kbtu_to_kwh
bpd_full['site_eui_kwh_m2'] = bpd_full['site_energy_kwh'] / bpd_full['floor_area_m2']
bpd_full['fuel_eui_kwh_m2'] = bpd_full['fuel_eui'] * kbtu_ft2_to_kwh_m2
bpd_full['electric_eui_kwh_m2'] = bpd_full['electric_eui'] * kbtu_ft2_to_kwh_m2
bpd_full['ghg_emissions_m2'] = bpd_full['ghg_emissions_int'] * bpd_full['floor_area'] / bpd_full['floor_area_m2']

# Some basic feature engineering
bpd_full['age'] = bpd_full['year']- bpd_full['year_built']
bpd_full['facility_type_full'] = bpd_full['facility_type']
bpd_full['facility_type'] = bpd_full['facility_type'].apply(lambda x: x.split(' - ')[0])

# %% [markdown]
# #### Feature engineering

# %%
# Additional feature engineering

# Ratio variables
bpd_full['electricity_fuel_ratio'] = bpd_full['electric_eui_kwh_m2'] / bpd_full['fuel_eui_kwh_m2']
bpd_full['electricity_fuel_ratio'] = bpd_full['electricity_fuel_ratio'].replace([np.inf, -np.inf], np.nan, inplace=False) # avoid infinity

# Interactions
bpd_full['age_floor_area'] = bpd_full['age'] * bpd_full['floor_area_m2']


# Nonlinearities
bpd_full['age_2'] = bpd_full['age'] ** 2
bpd_full['energy_star_rating_2'] = bpd_full['energy_star_rating'] ** 2
bpd_full['log_floor_area'] = np.log10(bpd_full['floor_area_m2'])

# store in a list 
engineered = ['electricity_fuel_ratio', 'age_floor_area', 'age_2', 'energy_star_rating_2', 'log_floor_area']

# %%
# General model
dum_cols = ['climate_code', 'facility_type']
data = model_data(bpd_full).restrict(area_max=999_999_999,
                                     area_min=40,
                                     eui_max=400,
                                     eui_min=15).dummify(columns=dum_cols).get_data()

# %%
data.id.unique().shape

# %%
bpd_full[engineered].describe()

# %%
# Imputation & scaling
# Select columns for preprocessing
columns_drop = ['city_x', 'building_class', 'floor_area','year_built','electric_eui','fuel_eui','site_eui','source_eui',
                'ghg_emissions_int', 'data_source','county_code', 'request_id','type', 'climate_trimmed','site_energy_kBTU',
                'site_energy_kwh', 'facility_type_full', 'state_x', 'year', 'state_y', 'city_y', 'place', 'latitude', 'longitude',
                'climate', 'county', 'fuel_eui_kwh_m2', 'electric_eui_kwh_m2', 'ghg_emissions_m2',
               ]
cols_to_keep = [c for c in data.columns if 'feature' in c] + ['id', 'site_eui_kwh_m2', 'zip_code']
cols_to_scale = [c for c in data.columns if c not in (columns_drop+cols_to_keep)] # numeric columns

# Define preprocessing for numeric columns
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),     # median imputation
    ("scaler", StandardScaler())                       # z-score scaling
])

# Combine transformations
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, cols_to_scale),
        ("keep", "passthrough", cols_to_keep)          # keep other columns unchanged
    ]
)

# Fit and transform
df_processed = preprocessor.fit_transform(data)

# Convert back to DataFrame with proper column names
df_processed = pd.DataFrame(df_processed, columns=cols_to_scale + cols_to_keep)

# %%
df_proc_2 = df_processed.copy().dropna().reset_index(drop=True)

# %%
# create all possible feature splits 
excluded_columns = ['id', 'site_eui_kwh_m2', 'latitude', 'longitude', 'zip_code']

# All available columns
all_x_columns = [c for c in df_proc_2.columns if c not in (excluded_columns)]

# build index list
col_to_idx = {name: i for i, name in enumerate(df_proc_2[all_x_columns].columns)} # need to be cast only on x cols as we must ignore index of columns not in use
all_x_idx = [col_to_idx[c] for c in all_x_columns] 

# Climate features
climate_x_cols = [c for c in df_proc_2.columns if re.search(r'\d+_', c)]
climate_x_idx = [col_to_idx[c] for c in climate_x_cols]

# Embeddings
embeddings_x_cols = [c for c in df_proc_2.columns if 'feature' in c]
embeddings_x_idx = [col_to_idx[c] for c in embeddings_x_cols]

# Baseline BPD features
baseline_x_cols = [c for c in df_proc_2.columns if c not in (climate_x_cols + embeddings_x_cols + engineered + excluded_columns)]
baseline_x_idx = [col_to_idx[c] for c in baseline_x_cols]

# Baseline and climate features
baseline_climate_x_cols = baseline_x_cols + climate_x_cols
baseline_climate_x_idx = [col_to_idx[c] for c in baseline_climate_x_cols]

# Baseline and embeddings
baseline_embeddings_x_cols = baseline_x_cols + embeddings_x_cols
baseline_embeddings_x_idx = [col_to_idx[c] for c in baseline_embeddings_x_cols]

# All x without feature engineered
all_noengineered_x_cols = [c for c in df_proc_2.columns if c not in (excluded_columns + engineered)]
all_noengineered_x_idx = [col_to_idx[c] for c in all_noengineered_x_cols]

# Baseline + engineered
baseline_engineered_x_cols = baseline_x_cols + engineered
baseline_engineered_x_idx = [col_to_idx[c] for c in baseline_engineered_x_cols]

# Baseline without energy and climate explicit (climate, energy star) – building characteristics
removed = ['id', 'site_eui_kwh_m2', 'energy_star_rating',  'climate_code_2B',
 'climate_code_3B',
 'climate_code_3C',
 'climate_code_4A',
 'climate_code_4B',
 'climate_code_4C',
 'climate_code_5A',
 'climate_code_5B' ]
baseline_noclimate_x_cols = [c for c in df_proc_2.columns if c not in (climate_x_cols + embeddings_x_cols + excluded_columns + engineered + removed)]
baseline_noclimate_x_idx = [col_to_idx[c] for c in baseline_noclimate_x_cols]

# %%
df_proc_2[all_x_columns +['site_eui_kwh_m2']] = df_proc_2[all_x_columns +['site_eui_kwh_m2']].astype("float32")
df_proc_2["id"] = df_proc_2["id"].astype(str)

# %%
# Cross validation splits

random.seed(3);cv_seeds = random.sample(range(1, 10_000), 5)

cv_splits = {}

for seed in cv_seeds:
    train, test = randomise_X_y_idx(df_proc_2, split=0.75, random_seed=seed)
    cv_splits[seed] = {'train_idx': train, 'test_idx': test}

# %%
# save climate columns for description purposes 
climate_cols_desc = [c.split("_", 1)[1] for c in climate_x_cols]
climate_cols_desc = pd.DataFrame(data = {'id':climate_cols_desc})[['id']].drop_duplicates(keep='first').reset_index().drop(columns=['index'])
climate_cols_desc = climate_cols_desc.merge(data_descriptions, on='id', how='left')

# add descriptions for columns with no data in NOAA API
fill_map = {
    'ADPT': 'Monthly Average Dew Point Temperature. Average of daily dew point temperatures given in Celsius or Fahrenheit depending on user specification. Missing if more than 5 days within the month are missing or flagged or if more than 3 consecutive values within the month are missing or flagged. DaysMissing: Flag indicating number of days missing or flagged (from 1 to 5).',
    'ASLP': 'Monthly Average Sea Level Pressure. Average of daily sea level pressures given in hPa or inches of mercury depending on user specification. Missing if more than 5 days within the month are missing or flagged or if more than 3 consecutive values within the month are missing or flagged. DaysMissing: Flag indicating number of days missing or flagged (from 1 to 5).',
    'ASTP': 'Monthly Average Station Level Pressure. Average of daily station level pressures given in hPa or inches of mercury depending on user specification. Missing if more than 5 days within the month aremissing or flagged or if more than 3 consecutive values within the month are missing or flagged.DaysMissing: Flag indicating number of days missing or flagged (from 1 to 5).',
    'AWBT': 'Monthly Average Wet Bulb Temperature. Average of daily wet bulb temperatures given in Celsius or Fahrenheit depending on user specification. Missing if more than 5 days within the month are missing or flagged or if more than 3 consecutive values within the month are missing or flagged. DaysMissing: Flag indicating number of days missing or flagged (from 1 to 5).',
    'DP1X': 'Number of days with >= 1.00 inch/25.4 millimeters in the month',
    'DYFG': 'Number of Days with Fog',
    'DYHF': 'Number of Days with Heavy Fog (visibility less than 1/4 statute mile)',
    'DYTS': 'Number of Days with Thunderstorms',
    'RHAV': 'Monthly Average Relative Humidity. Average of daily relative humidity values given in percent. Missing if more than 5 days within the month are missing or flagged or if more than 3 consecutive values within the month are missing or flagged. DaysMissing: Flag indicating number of days missing or flagged (from 1 to 5).',
    'RHMN': 'Monthly Average of Minimum Relative Humidity. Average of daily minimum relative humidity values given in percent. Missing if more than 5 days within the month are missing or flagged or if more than 3 consecutive values within the month are missing or flagged. DaysMissing: Flag indicating number of days missing or flagged (from 1 to 5).',
    'RHMX': 'Monthly Average of Maximum Relative Humidity. Average of daily maximum relative humidity values given in percent. Missing if more than 5 days within the month are missing or flagged or if more than 3 consecutive values within the month are missing or flagged. DaysMissing: Flag indicating number of days missing or flagged (from 1 to 5).'
}

climate_cols_desc['description'] = climate_cols_desc['description'].fillna(climate_cols_desc['id'].map(fill_map))

# add baseline columns
baseline = {
    "id": ['id', 'year', 'zip_code', 'city', 'state', 'climate', 'facility_type',
           'floor_area_m2', 'year_built','energy_star_rating', 'electric_eui', 'fuel_eui', 'site_eui', 'ghg_emissions_int',
           'population'],
    "description": ['Building ID','Year of data record', 'Postcode of the building', 'City of the building',
        'State of the building', 'ASHRAE climate code of the building', 'Build type of the building (e.g. single/multi family).',
        'Gross floor are in m2', 'Year in which the building was constructed', 'Energy star rating of the building',
        'EUI of electricity in kWh/m2/year', 'EUI of fuel use in kWh/m2/year', 'Total site EUI of the building in kWh/m2/year',
        'GHG emissions of the building', 'Population of the postcode area']
}
climate_cols_desc = pd.concat([climate_cols_desc, pd.DataFrame(baseline)], axis=0).reset_index(drop=True)

# %%
cols_desc_path = os.path.join(path,'outputs','model-tables','table-cols-descr.csv')
climate_cols_desc.to_csv(cols_desc_path, index=False)

# %% [markdown]
# ## 1) Find the best performing method

# %%
# Define baseline models and store them in an object
X_all = df_proc_2[all_x_columns].astype("float32").to_numpy()
y_all = df_proc_2['site_eui_kwh_m2'].astype("float32").to_numpy()

# Linear regression
lm_model = ml_model(linear_model.LinearRegression())

# Ridge regression
ridge_model = ml_model(linear_model.Ridge(alpha=1.0))

# Random forest
rfr_model = ml_model(RFR(random_state=3, n_jobs=6))

# XGB
xgb = XGBRegressor(random_state=3, learning_rate=0.3,n_jobs=-1,
                   eval_metric='rmse')
xgb_model = ml_model(xgb)

# ANN
nn = Sequential([
    Input(shape=(len(all_x_columns),)), 
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(1) # Output layer for regression
])

# Compile model
nn.compile(optimizer='adam', loss='mse', metrics=['mae'])

ann_model = ml_model(nn)

# Store models in an object 
models = {
    'ANN':ann_model,
    'Linear Regression':lm_model, 
    'Ridge Regression':ridge_model,
    'Random Forest':rfr_model, 
    'XGBoost':xgb_model,
    
}

# Prepare objects to store results
error_rates = {}
st_devs = {}

# Create a loop to train/evaluate models
for model_name, model in models.items():
    
    maes = []; rmses = []; r2_scores = []; mapes = []
    
    # Loop over cv_seeds
    for cv_seed in cv_seeds:

        model.load_data(all_x_idx, X_all, y_all, cv_splits[cv_seed]['train_idx'], cv_splits[cv_seed]['test_idx'])
        model.transformation()
        
        if model_name!='ANN':
            model.train()
        else:
            model.train(is_ann=True, validation_data=(model.X_test, model.y_test), epochs=15, batch_size=32, verbose=1)

        mae, rmse, r2, mape = model.predict_evaluate(return_data=True, verbose=False)
        maes.append(mae)
        rmses.append(rmse)
        r2_scores.append(r2)
        mapes.append(mape)

        
    # Store in the dictionary
    error_rates.update({model_name:{'MAE':np.array(maes).mean(),
                                    'RMSE':np.array(rmses).mean(),
                                    'R2':np.array(r2_scores).mean(),
                                    'MAPE':np.array(mapes).mean()}})

    # Store in the dictionary
    st_devs.update({model_name:{'MAE':np.array(maes).std(),
                                'RMSE':np.array(rmses).std(),
                                'R2':np.array(r2_scores).std(),
                                'MAPE':np.array(mapes).std()}})

# %%
st_devs

# %%
models = list(error_rates.keys())
metrics = list(next(iter(error_rates.values())).keys())

# classic colormap
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(len(models))]

metric_values = {metric: [error_rates[m][metric] for m in models] for metric in metrics}

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.flatten()

for ax, metric in zip(axes, metrics):
    bars = ax.bar(models, metric_values[metric], color=colors, alpha=0.9)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",          # format as needed
            ha='center',
            va='bottom',
            fontsize=9
        )

    ax.set_axisbelow(True)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha="right")

plt.tight_layout()
plt.savefig(os.path.join(path,'outputs','model-figures','figure-models-comparison.png')) 
plt.savefig(os.path.join(path,'outputs','model-figures','figure-models-comparison.eps'))
plt.show()

# %% [markdown]
# ### 1a) Compare against one-entry-per building model

# %%
# Prepare the data
one_building = df_proc_2[all_x_columns + ['id', 'site_eui_kwh_m2']].groupby(['id']).mean().reset_index()
ob_to_idx = {name: i for i, name in enumerate(one_building[all_x_columns].columns)} # need to be cast only on x cols as we must ignore index of columns not in use
ob_x_idx = [ob_to_idx[c] for c in all_x_columns] 

# X / y split
X_ob = one_building[all_x_columns].astype("float32").to_numpy()
y_ob = one_building['site_eui_kwh_m2'].astype("float32").to_numpy()

# XGB
xgb = XGBRegressor(random_state=3, learning_rate=0.3,n_jobs=-1,
                   eval_metric='rmse')
xgb_ob = ml_model(xgb)

# train / test splits
random.seed(200); ob_seed = random.sample(range(1, 10_000), 1)
train_ob, test_ob = randomise_X_y_idx(one_building, split=0.75, random_seed=ob_seed[0])

xgb_ob.load_data(ob_x_idx, X_ob, y_ob, train_ob, test_ob)
xgb_ob.transformation()
xgb_ob.train()
mae_ob, rmse_ob, r2_ob, mape_ob = xgb_ob.predict_evaluate(return_data=True, verbose=False)

# %% [markdown]
# ### 1b) Compare against model for predicting average EUI per zipcode

# %%
# Prepare the data
zipcode_embeddings = df_proc_2[embeddings_x_cols + ['zip_code', 'site_eui_kwh_m2']].groupby(['zip_code']).mean().reset_index()
em_to_idx = {name: i for i, name in enumerate(zipcode_embeddings[embeddings_x_cols].columns)} # need to be cast only on x cols as we must ignore index of columns not in use
em_x_idx = [em_to_idx[c] for c in embeddings_x_cols] 

# X / y split
X_ze = zipcode_embeddings[embeddings_x_cols].astype("float32").to_numpy()
y_ze = zipcode_embeddings['site_eui_kwh_m2'].astype("float32").to_numpy()

# XGB
xgb = XGBRegressor(random_state=3, learning_rate=0.3,n_jobs=-1,
                   eval_metric='rmse')
xgb_ze = ml_model(xgb)

# train / test splits
random.seed(200); ze_seed = random.sample(range(1, 10_000), 1)
train_ze, test_ze = randomise_X_y_idx(zipcode_embeddings, split=0.75, random_seed=ze_seed[0], id_column='zip_code')

xgb_ze.load_data(em_x_idx, X_ze, y_ze, train_ze, test_ze)
xgb_ze.transformation()
xgb_ze.train()
mae_ze, rmse_ze, r2_ze, mape_ze = xgb_ze.predict_evaluate(return_data=True, verbose=False)

# %% [markdown]
# ### Plot Comparisons

# %%
comparison_errors = {'Average per Building XGBoost':{
    'MAE':mae_ob,
    'RMSE':rmse_ob,
    'R2':r2_ob,
    'MAPE':mape_ob},
                    'Average Postcode EUI from embeddings XGBoost':{
    'MAE':mae_ze,
    'RMSE':rmse_ze,
    'R2':r2_ze,
    'MAPE':mape_ze}, 
              'Baseline XGBoost':error_rates['XGBoost']
}

comparison_names = list(comparison_errors.keys())
comparison_metrics = list(next(iter(comparison_errors.values())).keys())

# classic colormap
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(len(comparison_names))]

comparison_metric_values = {metric: [comparison_errors[m][metric] for m in comparison_names] for metric in comparison_metrics}

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for ax, metric in zip(axes, comparison_metrics):
    bars = ax.bar(comparison_names, comparison_metric_values[metric], color=colors, alpha=0.9)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    ax.set_axisbelow(True)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(comparison_names)))
    ax.set_xticklabels(comparison_names, rotation=15, ha="right")
    ax.margins(x=0.5)

plt.tight_layout()
plt.savefig(os.path.join(path,'outputs','model-figures','figure-experimental-comparison.png')) 
plt.savefig(os.path.join(path,'outputs','model-figures','figure-experimental-comparison.eps'))
plt.show()

# %% [markdown]
# ## 2) Fine-tune the best model

# %%
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import KFold


# %%
# Create train/test/validation split into a new object
random.seed(3); bayes_seed = random.sample(range(1, 10_000), 1)

bayes_split = {}
bayes_train_cv = {}

# split data into train / test
train_bayes, test_bayes = randomise_X_y_idx(df_proc_2, split=0.8, random_seed=bayes_seed[0])

# split training data into train / test splits for internal CV
random.seed(3); bayes_seeds = random.sample(range(1, 10_000), 3)

bayes_df = df_proc_2.iloc[train_bayes].reset_index(drop=True)

for seed in bayes_seeds:
    train, test = randomise_X_y_idx(bayes_df, split=0.8, random_seed=seed)
    bayes_train_cv[seed] = {'train_idx': train, 'test_idx': test}


# %%
# Translate object into array for BayesianCV
cv_bayes = [
    (bayes_train_cv[key]['train_idx'], bayes_train_cv[key]['test_idx'])
    for key in bayes_train_cv
]

# Prepare data
X_train_bayes = bayes_df[all_x_columns].astype("float32")
y_train_bayes = bayes_df['site_eui_kwh_m2'].astype("float32")

# Define new model
xgb = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=3,
        n_jobs=-1,
        eval_metric='rmse'
)

xgb_model = ml_model(xgb)

# Define Bayesian search space
search_space = {
    "max_depth": Integer(2, 12),
    "learning_rate": Real(0.001, 0.3, prior="log-uniform"),
    "n_estimators": Integer(100, 2000),
    "subsample": Real(0.5, 1.0),
    "colsample_bytree": Real(0.5, 1.0),
}


# Bayesian search
opt = BayesSearchCV(
    estimator=xgb,
    search_spaces=search_space,
    n_iter=10,               # number of Bayesian optimization steps
    cv=cv_bayes,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# Run optimization
opt.fit(X_train_bayes, y_train_bayes)

# %%
print("Best Score:", opt.best_score_)
print("Best Params:", opt.best_params_)
best_model = opt.best_estimator_

# %%
# Comparison chart 
xgb_tuned = XGBRegressor(
    **opt.best_params_,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42
)

xgb_tuned_model = ml_model(xgb_tuned)

X_all = df_proc_2[all_x_columns].astype("float32").to_numpy()

xgb_tuned_model.load_data(all_x_idx, X_all, y_all, train_bayes, test_bayes)
xgb_tuned_model.transformation()
xgb_tuned_model.train()
mae, rmse, r2, mape = xgb_tuned_model.predict_evaluate(return_data=True, verbose=False)

xgb_errors = {'Tuned XGBoost':{
    'MAE':mae,
    'RMSE':rmse,
    'R2':r2,
    'MAPE':mape},
              'Baseline XGBoost':error_rates['XGBoost']
}

# %%
xgb_names = list(xgb_errors.keys())
xgb_metrics = list(next(iter(xgb_errors.values())).keys())

# classic colormap
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(len(xgb_names))]

xgb_metric_values = {metric: [xgb_errors[m][metric] for m in xgb_names] for metric in xgb_metrics}

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.flatten()

for ax, metric in zip(axes, xgb_metrics):

    bars = ax.bar(xgb_names, xgb_metric_values[metric], color=colors, alpha=0.9)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha='center',
            va='bottom',
            fontsize=9
        )
    ax.set_axisbelow(True)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(xgb_names)))
    ax.set_xticklabels(xgb_names, rotation=15, ha="right")
    ax.margins(x=0.5)

plt.tight_layout()
plt.savefig(os.path.join(path,'outputs','model-figures','figure-tuned-comparison.png')) 
plt.savefig(os.path.join(path,'outputs','model-figures','figure-tuned-comparison.eps'))
plt.show()

# %% [markdown]
# ## 3) Identification of subsets predictive effects

# %%
# create objects to store the results 
subsets = {
    'Baseline features':{'columns': baseline_x_cols, 'idx': baseline_x_idx},
    'Weather features':{'columns':climate_x_cols, 'idx':climate_x_idx},
    'Embeddings features':{'columns':embeddings_x_cols, 'idx':embeddings_x_idx},
    'Baseline and Weather features':{'columns':baseline_climate_x_cols, 'idx':baseline_climate_x_idx},
    'Baseline and embeddings features':{'columns':baseline_embeddings_x_cols,'idx':baseline_embeddings_x_idx},
    'Baseline and engineered features':{'columns':baseline_engineered_x_cols,'idx':baseline_engineered_x_idx},
    'Baseline w/o weather and energy features':{'columns':baseline_noclimate_x_cols,'idx':baseline_noclimate_x_idx},
    'All w/o engineered features':{'columns':all_noengineered_x_cols,'idx':all_noengineered_x_idx},
    'All features':{'columns':all_x_columns, 'idx': all_x_idx}
}

subset_errors = {}
subset_importances = {}

# Define model
xgb_tuned = XGBRegressor(
    **opt.best_params_,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42
)

xgb_tuned_model = ml_model(xgb_tuned)


# Create a loop to train/evaluate models
for subset_name, subset in subsets.items():
    
    maes = []; rmses = []; r2_scores = []; mapes = []

    X_all = df_proc_2[all_x_columns].astype("float32").to_numpy()

    xgb_tuned_model.load_data(subset['idx'], X_all, y_all, train_bayes, test_bayes)
    xgb_tuned_model.transformation()
    xgb_tuned_model.train()
    mae, rmse, r2, mape = xgb_tuned_model.predict_evaluate(return_data=True, verbose=False)
    maes.append(mae)
    rmses.append(rmse)
    r2_scores.append(r2)
    mapes.append(mape)
        
    # Store in the dictionary
    subset_errors.update({subset_name:{'MAE':np.array(maes).mean(),
                                    'RMSE':np.array(rmses).mean(),
                                    'R2':np.array(r2_scores).mean(),
                                    'MAPE':np.array(mapes).mean()}})
    
    #  store feature importances 
    booster = xgb_tuned_model.model.get_booster()

    # Extract numeric importances
    importance_dict = booster.get_score(importance_type='gain')

    # Build a pandas series mapped to actual names
    feature_names = subset['columns']  # list of column names used in subset
    importances = pd.Series(0, index=feature_names, dtype=float)

    # Fill values
    for f, imp in importance_dict.items():
        idx = int(f[1:])      # "f23" -> 23
        importances.iloc[idx] = imp

    # Store only nonzero
    subset_importances[subset_name] = importances.sort_values(ascending=False)   

# %%
subset_names = list(subset_errors.keys())
subset_metrics = list(next(iter(subset_errors.values())).keys())

# classic colormap
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(len(subset_names))]

subset_metric_values = {metric: [subset_errors[m][metric] for m in subset_names] for metric in subset_metrics}

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for ax, metric in zip(axes, subset_metrics):
    bars = ax.bar(subset_names, subset_metric_values[metric], color=colors, alpha=0.9)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha='center',
            va='bottom',
            fontsize=9
        )
    ax.set_axisbelow(True)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(subset_names)))
    ax.set_xticklabels(subset_names, rotation=15, ha="right")

plt.tight_layout()
plt.savefig(os.path.join(path,'outputs','model-figures','figure-subsets-comparison.png')) 
plt.savefig(os.path.join(path,'outputs','model-figures','figure-subsets-comparison.eps'))
plt.show()

# %%
import math

# your dictionary
subset_names = list(subset_importances.keys())
n = len(subset_names)

# grid size (2 columns, auto rows)
rows = math.ceil(n / 2)
cols = 2 if n > 1 else 1

fig, axes = plt.subplots(rows, cols, figsize=(18, 22))
axes = axes.flatten() if n > 1 else [axes]

for ax, subset_name in zip(axes, subset_names):

    # get Series for this subset
    series = subset_importances[subset_name]

    # sort descending and optionally limit to top 20
    series_sorted = series.sort_values(ascending=False)[:20]

    ax.bar(series_sorted.index, series_sorted.values, alpha=0.9)
    ax.set_title('Feature importances of ' + subset_name)

    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    ax.set_xticks(range(len(series_sorted)))
    ax.set_xticklabels(series_sorted.index, rotation=25, ha="right", fontsize=9)

    ax.set_ylabel("Importance (gain)")

# hide unused axes if grid isn't full
for idx in range(len(subset_names), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(path,'outputs','model-figures','figure-feature-importances.png')) 
plt.savefig(os.path.join(path,'outputs','model-figures','figure-feature-importances.eps'))
plt.show()

# %% [markdown]
# ## Analysis of residuals and strong mispredictions with the tuned XGBoost model

# %% [markdown]
# ## XGBoost

# %%
xgb_tuned_model.load_data(all_x_idx, X_all, y_all, train_bayes, test_bayes)
xgb_tuned_model.transformation()
xgb_tuned_model.train()
xgb_tuned_model.predict_evaluate()

# %%
ax = xgb_tuned_model.plot_predictions('Site EUI (kWh/m2/yr)', 'Tuned XGBoost', save=True)

# %%
xgb_tuned_model.plot_residuals('Site EUI (kWh/m2/yr)', 'Tuned XGBoost', save=True)

# %%
# Compare groups with large residuals
resid_idx = (data.iloc[test_bayes,][baseline_engineered_x_cols]).copy()
resid_idx['observed'] = y_all[test_bayes]
resid_idx['predicted'] = xgb_tuned_model.y_pred
resid_idx['residuals'] = resid_idx['observed'] - resid_idx['predicted']
resid_idx = resid_idx[resid_idx['residuals'].abs()>=50]
resid_idx['positive_residual'] = np.where(resid_idx['residuals']>=0, True, False)

# %%
resid_path = os.path.join(path,'outputs','model-tables','table-residuals-analysis.csv')
resid_idx.groupby('positive_residual').describe().T.to_csv(resid_path)

# %% [markdown]
# ## Linear Regression

# %%
X_all = df_proc_2[all_x_columns].astype("float32").to_numpy()
y_all = df_proc_2['site_eui_kwh_m2'].astype("float32").to_numpy()

# %%
model = linear_model.LinearRegression()
lm = ml_model(model)

lm.load_data(all_x_idx, X_all, y_all, cv_splits[3899]['train_idx'], cv_splits[3899]['test_idx'])
lm.transformation()
lm.train()

# %%
lm.predict_evaluate()

# %%
lm.plot_predictions('site_eui_kwh_m2')

# %%
lm.plot_residuals('site_eui_kwh_m2')

# %% [markdown]
# ## Artificial Neural Network

# %%
# Build simple ANN model
nn = Sequential([
    Dense(512, activation='relu', input_shape=(len(all_x_columns),)),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(1) # Output layer for regression
])

# Compile model
nn.compile(optimizer='adam', loss='mse', metrics=['mae'])

nn_model = ml_model(nn)

nn_model.load_data(all_x_idx, X_all, y_all, cv_splits[3899]['train_idx'], cv_splits[3899]['test_idx'])
nn_model.transformation()
nn_model.train(is_ann=True, validation_data=(nn_model.X_test, nn_model.y_test), epochs=20, batch_size=32, verbose=1)

# Evaluate
loss, mae = nn_model.model.evaluate(nn_model.X_test, nn_model.y_test)
print(f"Test MAE: {mae:.4f}")

train_loss = nn_model.history.history['loss']
val_loss   = nn_model.history.history['val_loss']

# %%
plt.figure(figsize=(8,5))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# %%
nn_model.predict_evaluate() 

# %%
nn_model.plot_predictions('site_eui_kwh_m2')

# %%
nn_model.plot_residuals('site_eui_kwh_m2')

# %% [markdown]
# ## Ridge Regression

# %%
ridge_model = linear_model.Ridge(alpha=10)
ridge = ml_model(ridge_model)

ridge.load_data(all_x_idx, X_all, y_all, cv_splits[3899]['train_idx'], cv_splits[3899]['test_idx'])
ridge.transformation()
ridge.train()
ridge.predict_evaluate()

# %%
ridge.plot_predictions('site_eui_kwh_m2')

# %%
ridge.plot_residuals('site_eui_kwh_m2')

# %% [markdown]
# ## Vanilla Random Forest 

# %%
rfr = RFR(random_state=3, n_jobs=-1)
rf = ml_model(rfr)

rf.load_data(all_x_idx, X_all, y_all, cv_splits[3899]['train_idx'], cv_splits[3899]['test_idx'])
rf.transformation()
rf.train()

# %%
rf.predict_evaluate()

# %%
rf.plot_predictions('site_eui_kwh_m2')

# %%
rf.plot_residuals('site_eui_kwh_m2')

# %%
importances = pd.Series(rf.model.feature_importances_, index=rf.X_train.columns).sort_values(ascending=False)

importances.head(20).plot(kind='barh', figsize=(8, 10))
plt.title("Top 20 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Legacy Code

# %% [markdown]
# ## Simple NN

# %%
X_train, X_test, y_train, y_train_transformed, y_test, y_pred, y_pred_transformed = xgb.access_data()
y_train = y_train
y_test = y_test

# %%
train.columns[train.isna().any()].tolist()

# %%
np.isnan(X_train).any()

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build simple ANN model
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile model
nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
nn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=1)

# Evaluate
loss, mae = nn_model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.4f}")


# %%
y_pred = nn_model.predict(X_test)

# %%
print(f"The R2 score is: {r2_score(y_test, y_pred):,.2f}")
print(f"The RMSE score is: {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}\n")
print(f"The MAPE score is: {mean_absolute_percentage_error(y_test, y_pred)}\n")

# %%
fig, ax= plt.subplots()
ax.scatter(y_pred[:,0], y_test, s=5) ,

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
plt.set_cmap('Set1')
ax.plot(lims, lims, c="red")
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_title(f"Predicted vs Actual \n Site EUI")
ax.set_xlabel(f"Predicted Site EUI")
ax.set_ylabel(f"Actual Site EUI")
ax.legend()

# %%
y_pred[:,0]

# %%
residuals = y_test - y_pred[:,0]

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Residuals vs Predicted
sns.scatterplot(x=y_pred[:,0], y=residuals, ax=axs[0])
axs[0].axhline(0, color='red', linestyle='--')
axs[0].set_xlabel("Predicted values")
axs[0].set_ylabel("Residuals (y_true - y_pred)")
axs[0].set_title("Residuals vs Predicted")

# Residuals distribution
sns.histplot(residuals, bins=30, kde=True, ax=axs[1])
axs[1].set_xlabel("Residuals")
axs[1].set_title("Residuals Distribution")

plt.suptitle('')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Cross Validation of Performance

# %%
metrics_dict = {}

for seed, split in cv_splits.items():
    train = split['train']
    test = split['test']

    # Initialize model
    xgb_model = XGBRegressor(
        random_state=seed,
        learning_rate=0.3,
        n_jobs=-1,
        eval_metric='rmse'
    )

    # Train + evaluate
    xgb = ml_model(xgb_model)
    xgb.load_data(x_columns, 'site_eui_kwh_m2', train, test)
    xgb.transformation()
    xgb.train()
    xgb.predict_evaluate()

    # Store metrics
    r2 = r2_score(xgb.y_test, xgb.y_pred)
    rmse = np.sqrt(mean_squared_error(xgb.y_test, xgb.y_pred))
    mape = mean_absolute_percentage_error(xgb.y_test, xgb.y_pred)

    metrics_dict[seed] = {
        'R2': r2,
        'RMSE': rmse,
        'MAPE': mape
    }

# %%
metrics_dict

# %%
df = pd.DataFrame(metrics_dict).T  # transpose so seeds are rows
df = df.astype(float)               # ensure numeric types

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

df['R2'].plot(kind='bar', ax=axes[0], color='skyblue', legend=False)
axes[0].set_title('R² Comparison')
axes[0].set_ylabel('R²')
axes[0].set_xlabel('Seed')
axes[0].grid(axis='y', linestyle='--', alpha=0.6)

df['RMSE'].plot(kind='bar', ax=axes[1], color='salmon', legend=False)
axes[1].set_title('RMSE Comparison')
axes[1].set_ylabel('RMSE')
axes[1].set_xlabel('Seed')
axes[1].grid(axis='y', linestyle='--', alpha=0.6)

df['MAPE'].plot(kind='bar', ax=axes[2], color='lightgreen', legend=False)
axes[2].set_title('MAPE Comparison')
axes[2].set_ylabel('MAPE')
axes[2].set_xlabel('Seed')
axes[2].grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
