# %% [markdown]
# # Explore building data for NOAA weather data request
# Run tests to make the request as efficient as possible.

# %% [markdown]
# ## Imports
# %%
from time import sleep
import os
from wsgiref import headers 
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests
from datetime import datetime


def fetch_data(endpoint, headers, params=None, limit=1000):

    # get response
    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()
    return response


def get_count(endpoint, headers, params=None):
    attempt = 0
    wait_seconds = 1.5
    while True:
        try:
            count_params = params.copy() if params else {}
            count_params.update({"limit": 1})
            response = requests.get(endpoint, headers=headers, params=count_params)
            response.raise_for_status()
            return response.json()["metadata"]["resultset"]["count"]
        except requests.exceptions.RequestException as e:
            attempt += 1
            print(f"Error fetching count: {e}, attempt: {attempt}")
            sleep(min(wait_seconds * 2 ** attempt, 30))

def fetch_all_noaa_data(endpoint, headers, params=None, limit=1000, verbose=None):

    # get number of required requests and store offsets in a list
    n_requests = get_count(endpoint, headers, params=params)
    offsets = list(range(0, n_requests, limit))

    # prepare a list to store all data
    all_data = []

    # make a full request
    for offset in offsets:
        attempt = 0
        wait_seconds = 1.5
        try:
            offset_params = params.copy() if params else {}
            offset_params.update({"limit": limit, "offset": offset})
            response = fetch_data(endpoint, headers, params=offset_params)
            response.raise_for_status()
            data = response.json()["results"]  # adjust key depending on API
            all_data.extend(data)              # store in master list
            if verbose:
                print(f"Fetched {len(data)} entries from offset {offset}")
        except requests.exceptions.RequestException as e:
            attempt += 1
            print(f"Request failed at offset {offset}: {e}, attempt: {attempt},trying again...")
            sleep(min(wait_seconds * 2 ** attempt, 30))  # wait before retrying
        sleep(0.5)  # to avoid hitting rate limits

    # store in a df
    df = pd.DataFrame(all_data)
    return df


def process_noaa_data(df):
    if df.empty:
        return pd.DataFrame()

    df['month'] = pd.to_datetime(df['date']).dt.month
    grouped = df.groupby(['month', 'datatype'])['value'].mean().reset_index()
    # Pivot datatypes into columns
    pivoted = grouped.pivot(index='month', columns='datatype', values='value')
    # Rename to e.g., 01_TMIN, 02_TMIN
    df_flat = pd.DataFrame(pivoted.values.flatten()).T
    df_flat.columns = [f"{month}_{col}" for month in pivoted.index for col in pivoted.columns]
    return df_flat.T.rename(columns={0: 'value'})

# %%
# Load .env file if present
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'))

# Access token safely
NOAA_TOKEN = os.getenv("NOAA_TOKEN")

if not NOAA_TOKEN:
    raise ValueError("NOAA_TOKEN not found. Check your .env file or environment settings.")
else :
    print("NOAA_TOKEN loaded successfully.")


# Load building data
wd = os.getcwd()
# bpd_data = pd.read_csv(
    # os.path.dirname(os.path.dirname(wd)) + "/data/processed/bpd_data_processed.csv",
    # low_memory=False
    # )

# Load zipcode data
zip_data = pd.read_excel(
    os.path.dirname(os.path.dirname(wd)) + "/data/other/ZIP_COUNTY_062025.xlsx"
)

zip_data.columns = zip_data.columns.str.lower()
# %% [markdown]
# Data will be loaded in separate API calls – it could be done in one call but the data end up messy and hard to clearly distinquish
#  e.g. county name having a zipcode id etc.

# %%
# Load weather stations data
# fetch availabe zip locations from NOAA API
endpoint = "https://www.ncdc.noaa.gov/cdo-web/api/v2/locations"
params = { 
    "datasetid":"GSOM",
    "locationcategoryid":"ZIP"
}
headers = {"token": NOAA_TOKEN}
zip_locations = fetch_all_noaa_data(endpoint, headers=headers, params=params)

# %%
# fetch availabe city locations from NOAA API
endpoint = "https://www.ncdc.noaa.gov/cdo-web/api/v2/locations"
params = { 
    "datasetid":"GSOM",
    "locationcategoryid":"CITY"
}
headers = {"token": NOAA_TOKEN}
city_locations = fetch_all_noaa_data(endpoint, headers=headers, params=params)

# %%
# fetch availabe county locations from NOAA API
endpoint = "https://www.ncdc.noaa.gov/cdo-web/api/v2/locations"
params = { 
    "datasetid":"GSOM",
    "locationcategoryid":"CNTY"
}
headers = {"token": NOAA_TOKEN}
county_locations = fetch_all_noaa_data(endpoint, headers=headers, params=params)

# %%
# fetch availabe county locations from NOAA API
endpoint = "https://www.ncdc.noaa.gov/cdo-web/api/v2/locations"
params = { 
    "datasetid":"GSOM",
    "locationcategoryid":"ST"
}
headers = {"token": NOAA_TOKEN}
state_locations = fetch_all_noaa_data(endpoint, headers=headers, params=params)

# %% [markdown]
# Now that all locations are fetched from the API, we can filter them to only include those in the BPD dataset.
# Before that, the data needs to be processed to id strings. The assumption is that in case there is no ZIP or city code for the building, there will for sure be county or state data available. If there is no identificator, the data is dropped.

# %%
# Process zip & city locations
zip_locations[["type", "zipcode"]] = zip_locations["id"].str.split(":", expand=True)

# cities
city_locations_processed = city_locations[city_locations['name'].str.contains('US')].copy() # filter only US cities
city_locations_processed[["city", "state_country"]] = city_locations_processed["name"].str.split(",", expand=True)
city_locations_processed["state"] = city_locations_processed["state_country"].str[1:3]
# Find and replace washington data as it is faulty
mask = city_locations_processed["city"].str.contains('Washington')
city_locations_processed.loc[mask, "city"] = "Washington"
city_locations_processed.loc[mask, "state"] = "DC"

# counties
county_locations[['type', 'county_code']] = county_locations["id"].str.split(":", expand=True)

# states
state_to_postal = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

# Create a dictionary to map postal names to state names
postal_to_state = {v: k for k, v in state_to_postal.items()}

# Add a new column 'State Postal' to df_state_id using the dictionary mapping
state_locations['state'] = state_locations['name'].map(state_to_postal)
state_locations['state'] = np.where(state_locations['name']=='District of Columbia', 'DC',state_locations['state'])


# %%
# Get unqiue zipcodes, cities, counties and states from the BPD dataset
bpd_zipcodes = bpd_data['zip_code'].dropna().unique()
bpd_cities = bpd_data['city'].dropna().unique()
zip_data['zip'] = zip_data['zip'].astype(str)  # Ensure ZIP codes are strings with leading zeros
bpd_county_codes = bpd_data.merge(zip_data[['zip', 'county']], left_on='zip_code', right_on='zip', how='left')['county'].dropna().unique()
bpd_data['county_code'] = bpd_data.merge(zip_data[['zip', 'county']], left_on='zip_code', right_on='zip', how='left')['county'].astype("Int64").astype("string") # also add county codes to the main dataset
bpd_states = bpd_data['state'].dropna().unique()


# %% [markdown]
# Now that we have all the unique identifiers that are present in the BPD dataset, we can start to find matching locations.

# %%
matching_zip_locations = zip_locations[zip_locations['zipcode'].isin(bpd_zipcodes)].copy()
matching_zip_locations = matching_zip_locations[matching_zip_locations['datacoverage']==1].copy() # filter out locations with low data coverage
matching_city_locations = city_locations_processed[city_locations_processed['city'].isin(bpd_cities)].copy()
bpd_county_codes = bpd_county_codes.astype(int).astype(str)
matching_county_locations = county_locations[county_locations['county_code'].isin(bpd_county_codes)].copy()
matching_state_locations = state_locations[state_locations['state'].isin(bpd_states)].copy()

# %% [markdown]
# Another step that will limit errors is the date coverage – we must filter out unique combinations of dates for each dataframe

# %%
# First, create complete unique list of locations and years in the dataset
unique_zips_years = bpd_data.drop_duplicates(subset=['zip_code','year'])[['zip_code','year']]
unique_cities_years = bpd_data.drop_duplicates(subset=['city','year'])[['city','year']]
unique_counties_years = bpd_data.drop_duplicates(subset=['county_code','year'])[['county_code','year']]
unique_states_years = bpd_data.drop_duplicates(subset=['state','year'])[['state','year']]

# group and find min max years
unique_zips_years = unique_zips_years.groupby('zip_code')['year'].agg(['min', 'max']).reset_index()
unique_cities_years = unique_cities_years.groupby('city')['year'].agg(['min', 'max']).reset_index()
unique_counties_years = unique_counties_years.groupby('county_code')['year'].agg(['min', 'max']).reset_index()
unique_states_years = unique_states_years.groupby('state')['year'].agg(['min', 'max']).reset_index()

# now add back to the matching locations min and max years and filter the data
matching_zip_locations = matching_zip_locations.merge(unique_zips_years, left_on='zipcode', right_on='zip_code', how='left')
matching_city_locations = matching_city_locations.merge(unique_cities_years, left_on='city', right_on='city', how='left')
matching_county_locations = matching_county_locations.merge(unique_counties_years, left_on='county_code', right_on='county_code', how='left')
matching_state_locations = matching_state_locations.merge(unique_states_years, left_on='state', right_on='state', how='left')


# %%
matching_zip_locations = matching_zip_locations[
    ( pd.to_datetime(matching_zip_locations['mindate']) < pd.to_datetime(matching_zip_locations['min'].astype(str) + '-01-01') ) &
    ( pd.to_datetime(matching_zip_locations['maxdate']) > pd.to_datetime(matching_zip_locations['max'].astype(str) + '-01-01') )
    ]

matching_city_locations = matching_city_locations[
    ( pd.to_datetime(matching_city_locations['mindate']) < pd.to_datetime(matching_city_locations['min'].astype(str) + '-01-01') ) &
    ( pd.to_datetime(matching_city_locations['maxdate']) > pd.to_datetime(matching_city_locations['max'].astype(str) + '-01-01') )
    ]

matching_county_locations = matching_county_locations[
    ( pd.to_datetime(matching_county_locations['mindate']) < pd.to_datetime(matching_county_locations['min'].astype(str) + '-01-01') ) &
    ( pd.to_datetime(matching_county_locations['maxdate']) > pd.to_datetime(matching_county_locations['max'].astype(str) + '-01-01') )
    ]

matching_state_locations = matching_state_locations[
    ( pd.to_datetime(matching_state_locations['mindate']) < pd.to_datetime(matching_state_locations['min'].astype(str) + '-01-01') ) &
    ( pd.to_datetime(matching_state_locations['maxdate']) > pd.to_datetime(matching_state_locations['max'].astype(str) + '-01-01') )
    ]
# %% [markdown]
# The last step before making a request is to find a request type for each location, it's location id based on the type and the year for which we will collect the data.
# Then we will make a list of unique requests and we can make the request to NOAA API.

# %%
# Prepare request columns
bpd_data["request_type"] = None
bpd_data["request_id"] = None

# Find mathching zip codes
#zip_merge = bpd_data.merge(
#    matching_zip_locations[["zipcode", "id"]],
#    left_on="zip_code",
#   right_on="zipcode",
#  how="left",
#    suffixes=("", "_zip")
#)
#mask_zip = zip_merge["id_zip"].notna()
#zip_merge.loc[mask_zip, "request_type"] = "zip"
#zip_merge.loc[mask_zip, "request_id"] = zip_merge.loc[mask_zip, "id_zip"]

# Find matching cities
#no_zip = ~mask_zip
city_merge = bpd_data.merge(
    matching_city_locations[["city", "state", "id"]],
    on=["city", "state"],
    how="left",
    suffixes=("", "_city")
)
# mask_city = no_zip & & city_merge["id_city"].notna()
mask_city = city_merge['request_id'].isna() & city_merge["id_city"].notna()
city_merge.loc[mask_city, "request_type"] = "city"
city_merge.loc[mask_city, "request_id"] = city_merge.loc[mask_city, "id_city"]

# Find matching counties
no_city = city_merge["request_type"].isna()
county_merge = city_merge.merge(
    matching_county_locations[["county_code", "id"]],
    on="county_code",
    how="left",
    suffixes=("", "_county")
)
mask_county = no_city & county_merge["id_county"].notna()
county_merge.loc[mask_county, "request_type"] = "county"
county_merge.loc[mask_county, "request_id"] = county_merge.loc[mask_county, "id_county"]

# Find matching states
no_county = county_merge["request_type"].isna()
state_merge = county_merge.merge(
    matching_state_locations[["state", "id"]],
    on="state",
    how="left",
    suffixes=("", "_state")
)
mask_state = no_county & state_merge["id_state"].notna()
state_merge.loc[mask_state, "request_type"] = "state"
state_merge.loc[mask_state, "request_id"] = state_merge.loc[mask_state, "id_state"]

final_location_requests = state_merge[["id", "year", "zip_code", "city", "state", "request_type", "request_id"]].copy()

# %%
# Finally, we create a list of unique locations and years
request_list = final_location_requests.drop_duplicates(subset=['year','request_type', 'request_id'])
request_list = request_list.dropna(subset='request_id') # ensure no missing value

# Calculate number of unique requests
num_unique_requests = request_list.shape[0] # multiply by 12 for each month
time_needed = num_unique_requests * 6.5 # assuming 0.5 second per request
print(f"Number of unique requests to be made: {num_unique_requests}")
print(f"Estimated time needed (in seconds): {time_needed}")
print(f"Estimated time needed (in minutes): {time_needed/60:.2f}")
print(f"Estimated time needed (in hours): {time_needed/3600:.2f}")

# %%
# To calculate n_requests, we need to call for each row, get estimated number of entries and then divide by 1000 (limit),
# assume we take 0.5-1-3s per request

headers = {"token": NOAA_TOKEN}

for idx, row in request_list.iterrows():
    params = {
        "datasetid": "GSOM",
        #"datatypeid": "CLDD,HTDD,PRCP,SNOW,TAVG,TMAX,TMIN",
        "locationid": row["request_id"],
        "units": "metric",
        "startdate": f"{row['year']}-01-01",
        "enddate": f"{row['year']}-12-31"
    }
    print(f"Fetching {row['request_id']} for {row['year']}...")

    try:
        count = get_count(endpoint, headers, params)
        request_list.loc[idx, "count"] = count

    except Exception as e:
        print(f"Failed fetching {row['request_id']} ({row['year']}): {e}")
        continue

# %%
request_list['requests'] = np.ceil(count_requests['count'] / 1000)

# %%
#count_requests = pd.read_csv('/Users/dominikmecko/Desktop/Work.nosync/MSc Bath/MSc-Thesis/src/data-exploration/n_requests_counts.csv')
count_requests = request_list.copy()
# Show calculations
print(f"Total number of entries to be fetched {count_requests['count'].sum():,.0f}")
print(f"Total number of requests: {count_requests['requests'].sum():,.0f}")
print(f"Average number of requests: {count_requests['requests'].sum() / len(count_requests['requests']):,.0f}")
print(f"Maximum number of requests {count_requests['requests'].max():,.0f} in {count_requests.loc[count_requests['requests'].idxmax()][['city', 'year', 'request_id']]}")
print(f"Expected time to gather all requests {count_requests['requests'].sum() * 6.5 / 60:,.0f} minutes.")
print(f"Expected time to gather all requests {count_requests['requests'].sum() * 6.5 / 60 / 60:,.0f} hours.")
print(f"Expected time to gather all requests {count_requests['requests'].sum() * 6.5 / 60 / 60 / 3:,.0f} hours (paralellised).")

# %%
request_list['cumulative_sum'] = request_list['requests'].cumsum()
target1 = request_list['requests'].sum() / 3
target2 = 2* request_list['requests'].sum() / 3

# Get the first index where cumsum >= target
idx1 = request_list['cumulative_sum'].searchsorted(target1)
idx2 = request_list['cumulative_sum'].searchsorted(target2)

print(idx1, idx2)

# %%
request_list.to_csv('request_list.csv', index=False)

# %% [markdown]
# As we can see, it is not too terrible – the script can work for 3 days nonstop to gather all data points and we 
# can calculate particular data splits to ensure the script does not go above daily threshold for 
# the number of requests

# %%
endpoint = "https://www.ncdc.noaa.gov/cdo-web/api/v2/datatypes"
params = { 
    "datasetid":"GSOM"
}
headers = {"token": NOAA_TOKEN}

all_data_types = fetch_all_noaa_data(endpoint, headers=headers, params=params)

# %% [markdown]
# We can still select the weather variables, although this is not desirable. 
# The question is – if the data coverage is not good enough for zips, maybe it is better to continue only with cities etc
# it will also decrease the number of requests.


# %%
# --- Test loop ---
endpoint = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
headers = {"token": NOAA_TOKEN}

#equest_list = request_list.reset_index()

#test_request = request_list[:65]

test_request = missing.iloc[:,1:12]

# %%
start_time = datetime.now()

enriched_records = []
error_records = []

for _, row in test_request.iterrows():

    if _ % 25 == 0:

        if enriched_records:
            final_df = pd.concat(enriched_records, ignore_index=True)
            processed_final = final_df.pivot_table(
                index=['year', 'request_id'], 
                columns='index', 
                values='value'
            ).reset_index()
            print(f'Exporting… batch-{_}')
            processed_final.to_csv(os.path.join('api-outputs', f'batch-{_}-n3.csv'), index=False)

            enriched_records = []

    params = {
        "datasetid": "GSOM",
        #"datatypeid": "CLDD,HTDD,PRCP,SNOW,TAVG,TMAX,TMIN",
        "locationid": row["request_id"],
        "units": "metric",
        "startdate": f"{row['year']}-01-01",
        "enddate": f"{row['year']}-12-31"
    }
    print(f"Fetching {row['request_id']} for {row['year']}...({_+1}/{len(test_request)})")

    try:
        df_raw = fetch_all_noaa_data(endpoint, headers, params)

        df_processed = process_noaa_data(df_raw)
        if df_processed.empty:
            continue

        # Add identifiers
        df_processed.insert(0, "id", row["id"])
        df_processed.insert(1, "year", row["year"])
        df_processed.insert(2, "request_id", row["request_id"])

        enriched_records.append(df_processed.reset_index())

    except Exception as e:
        print(f"Failed fetching {row['request_id']} ({row['year']}): {e}")
        error_records.append({
            "request_id": row["request_id"],
            "year": row["year"],
            "error": str(e)
        })
        continue

# Save any remaining records
if enriched_records:
    final_df = pd.concat(enriched_records, ignore_index=True)
    processed_final = final_df.pivot_table(
        index=['year', 'request_id'], 
        columns='index', 
        values='value'
    ).reset_index()
    processed_final.to_csv(os.path.join('api-outputs', f'batch-final-n3.csv'), index=False)

end_time = datetime.now()
# %%
average_time_per_request = (end_time - start_time).total_seconds() / (test_request['count'].sum() / 1000)
print(f"Average time per one request is {average_time_per_request:,.2f} seconds per one request.") 
# %%
# Calculate remaining time
remaining_requests = test_request.iloc[74:]['count'].sum() / 1000
# Show calculations
print(f"Total number of entries to be fetched {test_request.iloc[74:]['count'].sum():,.0f}")
print(f"Total number of requests: {remaining_requests:,.0f}")
print(f"Average number of requests: {(test_request.iloc[74:]['count'].sum() / 1000) / len(test_request.iloc[74:]['count']):,.0f}")
print(f"Maximum number of requests {test_request.iloc[74:]['count'].max():,.0f} in {test_request.iloc[74:].loc[test_request.iloc[74:]['count'].idxmax()][['city', 'year', 'request_id']]}")
print(f"Expected time to gather all requests {(test_request.iloc[74:]['count'].sum() / 1000) * 6.5 / 60:,.0f} minutes.")
print(f"Expected time to gather all requests {(test_request.iloc[74:]['count'].sum() / 1000) * 6.5 / 60 / 60:,.0f} hours.")

# %%
# Process all batch files 
path = os.path.join(os.getcwd(), 'api-outputs')
files = [os.path.join(path, file) for file in os.listdir(path) if (file.endswith('.csv'))]

for i, file in enumerate(files):

    temp_df = pd.read_csv(file, low_memory=False)
    
    if i == 0:
        merged_outputs = temp_df
    else:
        merged_outputs = pd.concat([merged_outputs, temp_df], ignore_index=True)

# %%
# Merge df1 with df2 on both keys, keeping all df1 rows
bpd_data_merged = bpd_data.merge(
    final_location_requests[['id', 'year', 'request_id']],
    on=['id', 'year'],
    how='left',
    suffixes=('', '_new')
)

# Replace only where a new value exists
bpd_data_merged['request_id'] = bpd_data_merged['request_id_new'].combine_first(bpd_data_merged['request_id'])

# Clean up
bpd_data_merged = bpd_data_merged.drop(columns='request_id_new')

# %%
merged_outputs

# %%
# Now add data and export
bpd_data_merged = bpd_data_merged.merge(
    merged_outputs,
    on=['year', 'request_id'],
    how='left'
)

# %%
bpd_data_merged.to_csv('bpd-weather-enriched.csv', index=False)

# %%
# Fetch data descriptions
endpoint = "https://www.ncdc.noaa.gov/cdo-web/api/v2/datatypes"
params = { 
    "datasetid":"GSOM"
}
headers = {"token": NOAA_TOKEN}
data_descriptions = fetch_all_noaa_data(endpoint, headers=headers, params=params)

# %%
data_descriptions = data_descriptions[['id', 'name']]
data_descriptions = data_descriptions.rename(columns={'name':'description'})

# %%
data_descriptions.to_csv()

# %%
# LEGACY CODE FOR MISSING ENTRIES BELOW

# %%
len(merged_outputs)

# %%
request_list.shape

# %% [markdown]
# As we can see there are some missing entries so we need to find which ones

# %%
request_list['request_id'] = np.where(request_list['request_id']=='CITY:US060007', 'FIPS:06', request_list['request_id'])

# %%
merged_requests = pd.merge(request_list, merged_outputs, on=['year', 'request_id'], how='left', indicator=True)
missing = merged_requests[merged_requests['_merge'] == 'left_only']

# %%
missing

# %%
missing

# %%
matching_city_locations[matching_city_locations['id']=='CITY:US060007']

# %%
final_location_requests[final_location_requests['state']=='CA']

# %%
county_code_missing = bpd_data[bpd_data['id']=='6832897']['county_code'].iloc[1]

# %%
merged_outputs[(merged_outputs['year'].isin(['2019', '2020', '2021', 2022])) & (merged_outputs['request_id']=='FIPS:06')]
