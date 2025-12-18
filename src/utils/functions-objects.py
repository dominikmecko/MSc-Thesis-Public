# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Requirements

# %%
import os
import requests
import pandas as pd


# %% [markdown]
# ### NOAA API Calls

# %%
def noaa_api_call(url, api_key, params=None):
    """
    Makes a GET request to a given API endpoint.

    Args:
        url (str): API endpoint.
        api_key (str): API token for authorization.
        params (dict, optional): Query parameters.

    Returns:
        dict: JSON response content.
    """
    headers = {"token": api_key}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    
    return response.json()

endpoint = "https://www.ncdc.noaa.gov/cdo-web/api/v2/locations"

headers = {"token": api_key}

cities_list = []
offset = 0
stop_flag = False

while not stop_flag:
    response = requests.get(endpoint, headers=headers, params={"locationcategoryid": "ZIP","limit": 1000,"offset": offset,"sortfield": "name",
    "sortorder": "asc"})
    results = response.json()["results"]
    for result in results:
        city = result["name"]
        cities_list.append(city)
    if len(set(cities_list)) != len(cities_list):
        stop_flag = True
        cities_list = list(set(cities_list))
        break
    offset += 1000

df = pd.DataFrame(cities_list, columns=["City"])
print(df.head())



# %%

# %% [markdown]
# ### BPD Datasets Processing

# %%
def process_bdp_files():

    # go to folder
    path = os.path.dirname(os.path.dirname(os.getcwd())) + '/data/bpd-files'

    # read every file if it is csv
    all_files = [os.path.join(path, file) for file in os.listdir(path) if (file.endswith('.csv'))]

    # concatenate files
    for i, file in enumerate(all_files):
        if file.endswith('.csv'): 
            temp_df = pd.read_csv(file)
        else: 
            raise TypeError('File extension not supported.')
        if (i == 0):
            merged_df = temp_df
        elif (i > 0):
            merged_df = pd.concat([merged_df, temp_df], ignore_index=True)
        else:
            print('An unknown error occurred.')
    
    # remove duplicates
    merged_df = merged_df.drop_duplicates(subset=['id'])

    # export final file
    merged_df.to_csv(os.path.join(path, 'bpd_data_processed.csv'), index=False)
 

# %% [markdown]
# ### Adding Weather Data

# %%

# %% [markdown]
# ### Analysis Functions

# %%

# %% [markdown]
# ### Modelling

# %%

## LEGACY CODE BELOW - TO BE MOVED TO utils/functions-objects.py ##
# Define functions
def noaa_api_call(endpoint, api_key, params=None, limit=1000, max_requests=None):
    """
    Makes a GET request to a given API endpoint.

    Args:
        url (str): API endpoint.
        api_key (str): API token for authorization.
        params (dict, optional): Query parameters.

    Returns:
        dict: JSON response content.
    """
    headers = {"token": api_key}
    
    # get number of required requests and store offsets in a list   
    limit_parameters = params.copy() if params else {}
    limit_parameters.update({"limit": 1})
    initial_response = requests.get(endpoint, headers=headers, params=limit_parameters)
    initial_response.raise_for_status()
    n_requests = initial_response.json()["metadata"]["resultset"]["count"]
    offsets = list(range(0, n_requests, limit))

    # prepare a list to store all data
    all_data = []

    # make a full request
    for offset in offsets:
        try:
            response = requests.get(endpoint, headers=headers, params={"limit": limit, "offset": offset})
            response.raise_for_status()
            data = response.json()["results"]  # adjust key depending on API
            all_data.extend(data)              # store in master list
            print(f"Fetched {len(data)} entries from offset {offset}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed at offset {offset}: {e}, trying again...")
            sleep(2)  # wait before retrying
            try:
                response = requests.get(endpoint, headers=headers, params={"limit": limit, "offset": offset})
                response.raise_for_status()
                data = response.json()["results"]  # adjust key depending on API
                all_data.extend(data)              # store in master list
                print(f"Fetched {len(data)} entries from offset {offset}")
            except requests.exceptions.RequestException as e:
                print(f"Request failed at offset {offset}: {e}, skipping...")
        sleep(0.5)  # to avoid hitting rate limits
    
    # store in a df
    df = pd.DataFrame(all_data)
    return df
