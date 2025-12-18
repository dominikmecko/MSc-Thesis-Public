import os
import pandas as pd

def process_bdp_files():

    # go to folder
    folder = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'data' , 'bpd-files')

    # read every file if it is csv
    all_files = [os.path.join(folder, file) for file in os.listdir(folder) if (file.endswith('.csv'))]

    # concatenate files
    for i, file in enumerate(all_files):
        if file.endswith('.csv'): 
            temp_df = pd.read_csv(file, low_memory=False)
        else: 
            raise TypeError('File extension not supported.')
        if (i == 0):
            merged_df = temp_df
        elif (i > 0):
            merged_df = pd.concat([merged_df, temp_df], ignore_index=True)
        else:
            print('An unknown error occurred.')
    
    # remove duplicates
    merged_df = merged_df.drop_duplicates(subset=['id', 'year'])

    # keep only residential properties
    merged_df = merged_df[merged_df['building_class'].str.contains('Residential')]

    # export final file
    print("Exporting processed BPD data...")
    merged_df.to_csv(os.path.join(os.path.dirname(folder), 'processed' ,'bpd_data_processed.csv'), index=False)

process_bdp_files()