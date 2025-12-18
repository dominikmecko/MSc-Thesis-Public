# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
# Imports
import pandas as pd
import numpy as np
import os
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %%
# pandas settings
pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 1500)
pd.options.display.float_format = '{:,.4f}'.format


# %%
# Functions and classes
class Dataset:

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def analyse_data(self, selected_columns=None):
        if selected_columns is None:
            selected_columns = list(self.dataframe.columns)
        xray_columns = ['variable','vartype','q_NaN','p_NaN','count','mean','median','mode','std','minimum','maximum','unique']
        xray = pd.DataFrame(columns=xray_columns)

        for column in selected_columns:
            row = []
            variable = column
            vartype = self.dataframe[column].dtype
            q_NaN = round(self.dataframe[column].isna().sum(),2)
            p_NaN = "{:.2%}".format(self.dataframe[column].isna().mean())
            if (self.dataframe[column].isna().mean() <= 0.05) & (self.dataframe[column].dtype=='object'):
                p_NaN = "{:.2%}".format((self.dataframe[column].isin(['No Value','Unknown']).sum() / len(self.dataframe[column])))
            count = self.dataframe[column].count()
            if self.dataframe[column].notna().any():
                mean = self.dataframe[column].mean() if vartype!='O' else 'categorical variable'
                median = self.dataframe[column].median() if vartype!='O' else 'categorical variable'
                mode = self.dataframe[column].mode()[0]
                std = self.dataframe[column].std() if vartype!='O' else 'categorical variable'
                minimum = self.dataframe[column].min() if vartype!='O' else 'categorical variable'
                maximum = self.dataframe[column].max() if vartype!='O' else 'categorical variable'
                unique = self.dataframe[column].nunique()
            else:
                mean, median, mode, std, minimum, maximum, unique = [None]*7
            xray.loc[len(xray)] = [variable, vartype, q_NaN, p_NaN, count, mean, median, mode, std, minimum, maximum, unique]
        return xray

def plot_distribution(data, column, bins=30, figsize=(8, 5), color="skyblue", top_n=None, x_label=None):
    """
    Plot a histogram with KDE for numeric columns, or a frequency bar plot for categorical columns.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to plot.
    bins : int, optional
        Number of histogram bins (default=30).
    figsize : tuple, optional
        Figure size (default=(8, 5)).
    color : str, optional
        Color of the plot elements (default='skyblue').
    top_n : int, optional
        Show only the top N categories for categorical columns (default: all).
    """
    plt.figure(figsize=figsize)
    
    col_data = data[column].dropna()
    
    if pd.api.types.is_numeric_dtype(col_data):
        # Numeric: histogram + KDE
        ax = sns.histplot(
            col_data,
            bins=bins,
            kde=True,
            color=color,
            line_kws={"lw": 2}
        )
        if ax.lines:
            ax.lines[0].set_color('crimson')
        plt.ylabel("Frequency", fontsize=12)
    
    else:
        # Categorical: bar plot of value counts
        counts = col_data.value_counts().sort_values(ascending=False)
        if top_n:
            counts = counts.head(top_n)
        sns.barplot(x=counts.index, y=counts.values, hue=col_data.unique()[:top_n], palette="pastel", legend=False)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha='right')
    
    plt.title(f"Distribution of {column}", fontsize=14)
    if x_label:
        plt.xlabel(x_label, fontsize=12)
    else:
        plt.xlabel(column, fontsize=12)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def run_pca(df, numeric_cols=None, n_components=2, plot_variance=True, plot_components=True):
    """
    Perform PCA on numeric columns of a DataFrame and visualize results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numeric_cols : list, optional
        Columns to include in PCA. If None, automatically select numeric columns.
    n_components : int, optional
        Number of principal components to compute (default=2).
    plot_variance : bool, optional
        Whether to plot explained variance ratio.
    plot_components : bool, optional
        Whether to plot first two principal components scatter.
    
    Returns
    -------
    pca_result : pd.DataFrame
        DataFrame of principal components.
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model.
    """
    # Select numeric columns
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    data = df[numeric_cols].dropna()
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)
    
    # Create DataFrame with principal components
    pca_result = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(n_components)])
    
    # Explained variance plot
    if plot_variance:
        plt.figure(figsize=(8,5))
        sns.barplot(x=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                    y=pca.explained_variance_ratio_,
                    color='skyblue')
        plt.ylabel("Explained Variance Ratio")
        plt.xlabel("Principal Components")
        plt.title("PCA Explained Variance")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.show()
    
    # Scatter plot of first two PCs
    if plot_components and n_components >= 2:
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=pca_result['PC1'], y=pca_result['PC2'])
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA: First Two Principal Components")
        plt.grid(alpha=0.3)
        plt.show()
    
    return pca_result, pca

def pca_biplot(pca_result, pca_model, feature_names, scale=1):
    """
    Scatter plot of first two PCs with feature loadings as arrows.
    
    Parameters
    ----------
    pca_result : pd.DataFrame
        PCA-transformed data with at least PC1 and PC2.
    pca_model : PCA object
        Fitted sklearn PCA model.
    feature_names : list
        List of original feature names corresponding to PCA.
    scale : float
        Scaling factor for arrows.
    """
    plt.figure(figsize=(8,6))
    
    # Scatter plot of points
    sns.scatterplot(x=pca_result['PC1'], y=pca_result['PC2'], alpha=0.5)
    
    # Arrows for feature loadings
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, 
                  pca_model.components_[0, i]*scale, 
                  pca_model.components_[1, i]*scale, 
                  color='red', alpha=0.7, head_width=0.05)
        plt.text(pca_model.components_[0, i]*scale*1.15,
                 pca_model.components_[1, i]*scale*1.15,
                 feature, color='black', ha='center', va='center', fontsize=10)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Biplot')
    plt.grid(alpha=0.3)
    plt.show()

# Constants
sq_ft_to_sq_m = 0.09290304
kbtu_to_kwh = 0.29307107017
ft_to_m = 0.3048
kbtu_ft2_to_kwh_m2 = 3.15459

# %%
# Data pre-engineering

# Load data
chunks = pd.read_csv('bpd-weather-enriched.csv', chunksize=100_000)
bpd_raw = pd.concat(chunks, ignore_index=True)

# Load embeddings
path = os.getcwd()
embeddings_path = os.path.join(os.path.dirname(os.path.dirname(path)), 'data', 'embeddings','zcta_embeddings.csv')
embeddings = pd.read_csv(embeddings_path)

# %%
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

# %%
# Pre-engineering steps to allow string values behave as missing values
bpd_full = bpd_full.replace(['No Value', 'Unknown'], np.nan)

# %%
bpd_final = Dataset(bpd_full.copy())

# %% [markdown]
# ### Data Structure Check

# %% [markdown]
# The first step is to check the structure of the data – missing values, data types and basic descriptive statistics…

# %%
# Descriptive analysis of the dataset to check for missing values and distributions
bpd_final.analyse_data()

# %% [markdown]
# There are a lot of indicative steps to be taken based on the analasis of the data. The most important points are:
# * lots of features have a high share of missing values, the highest shares will be filtered out
# * some features need conversion of data types
# * several categorical variables are available, some need some feature engineering
# * U.S. unit variables will be translated into European ones

# %%
# Filter missing values for the target column
bpd_full = bpd_full[bpd_full.site_eui.notna()]

# Filter columns with missing values above an arbitrary threshold
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

# Add dummy variables for all columns that need analysis
dum_cols = ['state_x', 'climate', 'facility_type']
bpd_dummified = pd.get_dummies(bpd_full, columns=dum_cols, drop_first=True)

# %%
# Store columns for further use
# create all possible feature splits 
excluded_columns = ['id', 'site_eui_kwh_m2', 'latitude', 'longitude', 'zip_code']

# All available columns
all_x_columns = [c for c in df_proc_2.columns if c not in (excluded_columns)]

# Climate features
climate_x_cols = [c for c in df_proc_2.columns if re.search(r'\d+_', c)]

# Embeddings
embeddings_x_cols = [c for c in df_proc_2.columns if 'feature' in c]

# Baseline BPD features
baseline_x_cols = [c for c in df_proc_2.columns if c not in (climate_x_cols + embeddings_x_cols + engineered + excluded_columns)]

# All x without feature engineered
all_noengineered_x_cols = [c for c in df_proc_2.columns if c not in (excluded_columns + engineered)]
all_noengineered_x_idx = [col_to_idx[c] for c in all_noengineered_x_cols]

# Baseline + engineered
baseline_engineered_x_cols = baseline_x_cols + engineered
baseline_engineered_x_idx = [col_to_idx[c] for c in baseline_engineered_x_cols]

# Baseline without energy and climate explicit (climate, energy star) – building characteristics
removed = ['id', 'site_eui_kwh_m2', 'energy_star',  'climate_code_2B',
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
# create a data object with the transformed dataset
bpd_final_2 = Dataset(bpd_full.copy())
# release the old object from memory
del bpd_final

# %% [markdown]
# ## Strategy
# The strategy is :
# * Structure –
#     * undestrand data structure
#     * remove missing values and identify outliers –> obvious ones can be removed here
#     * see distributions of important variables
#     * find columns with low variation
# * Relationships –
#     * plot relationships among variables
#     * check correlations
#     * PCA and linear dependence
# * Transformations and hidden patterns –
#     * clustering analysis
#     * see relationships and distributions of transformed variables

# %% [markdown]
# ### What is the data structure and descriptive statistics? 

# %% [markdown]
# Now, let's repeat the same check to see how if the structure improved.

# %%
bpd_final_2.analyse_data(selected_columns=bpd_final_2.dataframe.loc[:, ~bpd_final_2.dataframe.columns.str.contains('feature', case=False)])

# %% [markdown]
# ### Commentary
# We can see the structure is much better. The share of missing values dropped, and rows of missing values are now ready for a handling strategy (imputation, masking, deletion,…). Data types also look correct and we have several newly created features available too. Some distributional observations are readily available too:
# * the data span from 2010 to 2023  – this creates possibility for some useful time-series feature handling!
# * there will be some outliers in the target **site_eui_kwh_m2** as its minimum and maximum values are almost impossible to reach in real life and are likely a result of measurement / input error

# %%
# create descriptive cols for further analysis
desc_cols = ['site_eui_kwh_m2', 'facility_type','year_built','floor_area_m2',
             'energy_star_rating', 'climate_trimmed', 'electric_eui_kwh_m2', 'fuel_eui_kwh_m2',
             'ghg_emissions_m2', 'population']

# %%
bpd_final_2.data[desc_cols].describe()

# %%
# Prepare unique buildings dataset – for some features are fixed over time,
# and t-entries vary so we need to group to not distort distributions
unique_buildings = bpd_final_2.dataframe.drop_duplicates(subset="id")

# %% [markdown]
# ### Are there any outliers?
# An important step is to check for outliers of our target. Other variables can be important too but there is no capacity to check for all 560 features.

# %%
bpd_final_2.dataframe

# %%
# Boxplot of the target
ax = sns.boxplot(bpd_final_2.dataframe[['site_eui_kwh_m2']], width=0.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# As the boxplot hints, there are many values that are above 500 kWh/m<sup>2</sup>/year and are considered outliers. The sheer volume of this data is hinting at a potential data error due to measurement or incorrect input. It could be that these values had an incorrect decimal point in the original value in kBTU, which resulted in an erroneous entry. This is supported in practical evidence, where a typical residential building has  EUI of around 250 kWh/m<sup>2</sup>/year. Let's randomly pick 10 buildings witch such values and examine whether other entries (if there are more), somehow differ. We can validate with other features and/or look at building in the vicinity (e.g. by zipcode).

# %%
# pick random 5 buildings with outlier values and display values of 5 random buildings with the same zipcode or city
outliers_ids = unique_buildings[unique_buildings['site_eui_kwh_m2']>=500].sample(5, random_state=4)[['id', 'zip_code', 'city_x']]
random_zip_ids = unique_buildings[unique_buildings['zip_code'].isin(outliers_ids['zip_code'])].sample(5, random_state=4)['id']

# %%
# pick random 5 buildings with outlier values and display values of 5 random buildings with the same zipcode or city
bpd_final_2.dataframe[bpd_final_2.dataframe['id'].isin(outliers_ids['id'])][['id', 'year', 'city_x', 'site_eui']+ desc_cols]

# %%
# pick random 5 buildings with outlier values and display values of 5 random buildings with the same zipcode or city
bpd_final_2.dataframe[bpd_final_2.dataframe['id'].isin(random_zip_ids)][['id', 'year', 'city_x', 'site_eui']+ desc_cols]

# %% [markdown]
# ### Check distributions & Representativeness

# %%
# %config InlineBackend.figure_format = 'svg'

# %% [markdown]
# ### Site EUI

# %%
# Target
plot_distribution(bpd_final_2.dataframe, 'site_eui_kwh_m2', bins=50)


# %% [markdown]
# ### Commentary
# The distribution is clearly not normal – it resembles a power distribution, but it is likely log-normal, so let's transform the data with log<sub>10</sub> transformation.

# %%
# Log-normal
plot_distribution(np.log10(bpd_final_2.dataframe[['site_eui_kwh_m2']]), 'site_eui_kwh_m2', bins=50, x_label='log(site_eui_kwh_m2)')

# %% [markdown]
# ### Commentary
# The transformation shows a bimodial distribution, likely caused by a mixture of two distributions. It is possible that there are two populations — high and low energy-intensive buildings — and we need to uncover the latent variables that can help us distinguish between them if it is possible. This will probably cause problems with prediction, depending on which variables have the strongest effect. If there are two populations with two distinct effects of e.g. <code>floor_area_m2</code>, then we will see erroneous predictions. Let's test where these distributions differ. Although there is an overlap, let's analyse some descriptive statistics.

# %%
Dataset(bpd_final_2.dataframe[np.log10(bpd_final_2.dataframe['site_eui_kwh_m2'])<2.3][desc_cols]).analyse_data()

# %%
Dataset(bpd_final_2.dataframe[np.log10(bpd_final_2.dataframe['site_eui_kwh_m2'])>2.3][desc_cols]).analyse_data()

# %%
# plot distributions of log(EUI) based on categorical assignment 

# %% [markdown]
# ### Total Energy 

# %%
plot_distribution(bpd_final_2.dataframe, 'site_energy_kwh', bins=100)

# %%
# Log-normal
plot_distribution(np.log10(bpd_final_2.dataframe[['site_energy_kwh']]), 'site_energy_kwh', bins=50, x_label='log(site_energy_kwh)')

# %%
Dataset(bpd_final_2.dataframe[np.log10(bpd_final_2.dataframe['site_energy_kwh'])<5][desc_cols + ['site_energy_kwh']]).analyse_data()

# %%
Dataset(bpd_final_2.dataframe[np.log10(bpd_final_2.dataframe['site_energy_kwh'])>=5][desc_cols + ['site_energy_kwh']]).analyse_data()

# %% [markdown]
# ### Floor Area

# %%
# Floor area
plot_distribution(unique_buildings, 'floor_area_m2', bins=50)

# %%
plot_distribution(np.log10(unique_buildings[['floor_area_m2']]), 'floor_area_m2', bins=50, x_label='log10(floor_area_m2)')

# %% [markdown]
# ### Commentary
# The same effect is even more present in <code>floor_area</code>. The transformation shows distinct a bimodial distribution, probably in the same way as for <code>site_eui_kwh_m2</code>. Let's see if the grouping by <code>facility_type</code> reveals any more insights.

# %%
# code
Dataset(unique_buildings[np.log10(unique_buildings['floor_area_m2'])<3][desc_cols]).analyse_data()

# %%
# code
Dataset(unique_buildings[np.log10(unique_buildings['floor_area_m2'])>3][desc_cols]).analyse_data()

# %%
# %config InlineBackend.figure_format = 'png'

# %%
ax = sns.scatterplot(x=bpd_final_2.dataframe[bpd_final_2.dataframe['site_eui_kwh_m2']<1000]['floor_area_m2'],
                     y=bpd_final_2.dataframe[bpd_final_2.dataframe['site_eui_kwh_m2']<1000]['site_eui_kwh_m2'])

# %%
ax = sns.scatterplot(x=np.log10(bpd_final_2.dataframe['floor_area_m2']), y=bpd_final_2.dataframe['site_eui_kwh_m2'])

# %%
ax = sns.scatterplot(x=bpd_final_2.dataframe['floor_area_m2'], y=np.log10(bpd_final_2.dataframe['site_eui_kwh_m2']))

# %%
ax = sns.scatterplot(x=np.log10(bpd_final_2.dataframe['floor_area_m2']), y=np.log10(bpd_final_2.dataframe['site_eui_kwh_m2']))

# %%
ax = sns.scatterplot(x=np.log10(bpd_final_2.dataframe['floor_area_m2']), y=np.log10(bpd_final_2.dataframe['site_energy_kwh']))

# %% [markdown]
# ### Year Built

# %%
# Year built
plot_distribution(unique_buildings, 'year_built', bins=50)

# %%
# Age
plot_distribution(bpd_final_2.dataframe, 'age', bins=50)

# %% [markdown]
# ### Commentary
# <code>Age</code> and <code>year_built</code> are almost identical reversed variables. While <code>year_built</code> stays fixed, <code>age</code> takes a value of <code>year</code> - <code>year_built</code>. The time value is not too significant to change the distribution. We can see a bimodial distribution again, this time caused by a boom of building in-between the war period. The age structure of the data can indicate and help predict some variance in EUI, but without the information on the building material and renovation efforts, the prediction power of this variable is limited. We can test this assumption with a simple scatter plot.

# %%
# scatter plot of age and EUI
ax = sns.scatterplot(x=bpd_final_2.dataframe['age'], y=bpd_final_2.dataframe['site_eui_kwh_m2'])

# %% [markdown]
# ### Population

# %%
# Population
plot_distribution(unique_buildings, 'population', bins=50)

# %% [markdown]
# ### Commentary
# The <code>population</code> distribution is quite scattered but almost forming something resembling a uniform distribution (if we ignore some spikes). However, there is little expectation that population severly affects EUI. Although there might be some heat gains from densely populated areas, these are probably marginal without knowing all latent variables that impact the EUI. 

# %% [markdown]
# ### Energy Star Rating

# %%
# Energy star rating
plot_distribution(bpd_final_2.dataframe, 'energy_star_rating', bins=50)

# %% [markdown]
# ### Commentary
# The distribution of <code>energy_star_rating</code>, a type of voluntary benchmarking of the U.S. building stock reveals interesting insights. Since the purpose of the score is to benchmark the building against comparable peers, one could expect a normal distribution, if the benchmarking goal is to rank in the population. However, it could follow another distribution, if the goal is to simply compare the performance against a static benchmark that does not depend on a dynamic variable in a given category. We see that:
# * There are big spikes on 0 and 100
#     * 0-spike just means missing data
#     * 100-spike is more *interesting* since it points to a sampling bias – better scoring buildings are more likely to be in the database, which is a data problem
# * The score linearly increases with the density. This can also be a sampling bias but it can be a methodological flaw in the rating's design.

# %% [markdown]
# ### Climates

# %%
# Climates
plot_distribution(unique_buildings, 'climate_trimmed')

# %%
plot_distribution(unique_buildings, 'facility_type_full', figsize=(8,6) ,top_n=7)

# %% [markdown]
# ### Commentary
#

# %% [markdown]
# Facility types and climates indicate that the sample isn't representative, especially by <code>facility_type</code>. The literature review has shown that the majority of assets are single homes in the US. However, these can be hidden under <code>Residential - Other</code>, because some studies claim that the BPD dataset is representative of the building stock.

# %%
# States
plot_distribution(unique_buildings, 'state_x')


# %%
# Cities
plot_distribution(unique_buildings, 'city_x', top_n=10)

# %% [markdown]
# ### Commentary
# The same can be seen in the distribution of cities and states. Majority of buildings are located in Gainesville, Florida. This is a distinct climate, so the external validity can be affected by the model being too reliant on hot-humid environment conditions. This can be remedied as the sample is quite large and also through dummy variables.

# %% [markdown]
# ## Principal Components Analysis

# %%
# Run pca analysis through the function defined above
pca_result, pca_model = run_pca(bpd_final_2.dataframe, n_components=3, plot_variance=True, plot_components=True)

# %%
loadings = pd.DataFrame(pca_model.components_.T, 
                        columns=[f'PC{i+1}' for i in range(pca_model.n_components_)],
                        index=bpd_final_2.dataframe.select_dtypes(include=np.number).columns.tolist())

# %%
# Absolute value to find strongest contributors
top_n = 15  # number of top features per PC
for pc in loadings.columns:
    print(f"\nTop features contributing to {pc}:")
    print(loadings[pc].abs().sort_values(ascending=False).head(top_n))

# %%
# Compute loading magnitudes
loading_magnitude = np.sqrt(pca_model.components_[0,:]**2 + pca_model.components_[1,:]**2)

# Pick top N features
top_n = 50
top_features_idx = np.argsort(loading_magnitude)[-top_n:]
top_features = [bpd_final_2.dataframe.select_dtypes(include=np.number).columns.tolist()[i] for i in top_features_idx]

# Plot biplot with only these arrows
pca_biplot(pca_result, pca_model, top_features, scale=150)

# %% [markdown]
# ### Commentary

# %% [markdown]
# The PCA Analysis shows very strong clustering based on the climate variables. The biplot shows a handful of distinct clusters and underlying variables. This means that the predictive model might benefit from the reduction of weather features as they're linearly dependent. Just two principal components can reduce the number of features significantly, reducing overfitting and computation demands. However, it might be interesting to see PCA on variables other than weather features.

# %% [markdown]
# ### Cluster Analysis

# %%

# %% [markdown]
# ### Numerical relationships (scatter plots)

# %%
# %config InlineBackend.figure_format = 'png'
cols = ['site_eui_kwh_m2','site_energy_kwh','floor_area_m2','energy_star_rating','year_built']#+['sum_CLDD','sum_HTDD','sum_PRCP','sum_SNOW']
sns.pairplot(bpd_final_2.dataframe[cols]);

# %% [markdown]
# ### Commentary
# Pairwise scatter plots do not reveal significant or unexpected relationships. Also, obvious outliers are destroying the scale of some charts. 

# %%
# %config InlineBackend.figure_format = 'svg'

# %%
corr_cols = bpd_final_2.dataframe.loc[:, ~bpd_final_2.dataframe.columns.str.contains('feature', case=False)].select_dtypes(include=np.number).columns.tolist()
corr_drop = ['year', 'zip_code', 'county_code', 'latitude', 'longitude']
corr_cols = [e for e in corr_cols if e not in corr_drop]
corr_matrix = bpd_final_2.dataframe[corr_cols].corr()

# %%
corr_matrix[abs(corr_matrix['site_eui_kwh_m2'])>=0.3][['site_eui_kwh_m2']].sort_values('site_eui_kwh_m2', ascending=False)

# %% [markdown]
# ### Commentary 
# There are some interesting correlations. Apart from the obvious variables like <code>fuel_eui</code> and <code>ghg_emissions
# </code>, several climate variables, also depending on the month are strongly to moderately correlated with <code>
# site_eui</code>, such as heating and cooling degree days, temperature, wind, precipitation and running sum variables (e.g. heating degree days to date).

# %%
ax = sns.heatmap(corr_matrix[['site_eui_kwh_m2']])

# %% [markdown]
# ### Analyse EUI across states

# %%
bpd_final_2.dataframe.groupby('state_x')['site_eui_kwh_m2'].describe().map(lambda x: f"{x:.2f}")

# %%
bpd_final_2.dataframe.groupby('climate_trimmed')['site_eui_kwh_m2'].describe().map(lambda x: f"{x:.2f}")

# %%
counties = bpd_final_2.dataframe.groupby(['county','state_x','climate_trimmed'])['site_eui_kwh_m2'].describe().sort_values(by='state_x')
counties[counties['count']>=200].map(lambda x: f"{x:.2f}")

# %%
cities = bpd_final_2.dataframe.groupby(['city_x','climate_trimmed'])['site_eui_kwh_m2'].describe()
cities[cities['count']>=500].map(lambda x: f"{x:.2f}")

# %% [markdown]
# ### Map with energy intensity

# %%
import folium
from folium import CircleMarker

# Example: assume df has columns ['latitude', 'longitude', 'site_energy_eui_kwh_m2']
map_data = bpd_final_2.dataframe[bpd_final_2.dataframe['site_eui_kwh_m2']<=500]

# 1. Aggregate data by coordinates
agg = (
    map_data.groupby(['latitude', 'longitude'], as_index=False)
      ['site_eui_kwh_m2']
      .mean()
)

# 2. Initialize map centered on your data
m = folium.Map(location=[agg['latitude'].mean(), agg['longitude'].mean()],
               zoom_start=6, tiles='CartoDB positron')

# 3. Normalize color scale
min_val, max_val = agg['site_eui_kwh_m2'].min(), agg['site_eui_kwh_m2'].max()

def color_scale(val):
    """Convert value to color on blue–red scale."""
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    norm = colors.Normalize(vmin=min_val, vmax=max_val)
    cmap = cm.get_cmap('RdYlGn_r')  # red = high EUI, green = low
    rgb = cmap(norm(val))[:3]
    return f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'

# 4. Add circle markers
for _, row in agg.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=6,
        color=color_scale(row['site_eui_kwh_m2']),
        fill=True,
        fill_color=color_scale(row['site_eui_kwh_m2']),
        fill_opacity=0.7,
        popup=folium.Popup(f"EUI: {row['site_eui_kwh_m2']:.1f} kWh/m²", max_width=150)
    ).add_to(m)

# 5. Add legend (optional)
from branca.colormap import LinearColormap
colormap = LinearColormap(
    colors=['green', 'yellow', 'red'],
    vmin=min_val,
    vmax=max_val,
    caption='Average site_eui_kwh_m2'
)
colormap.add_to(m)

m


# %%
bpd_final_2.dataframe.shape
