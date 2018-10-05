import pandas as pd 
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import numpy as np
import seaborn as sns

# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


def remove_outliers(df, trip_dur_max, trip_dur_min, dist_min):
    #delete select rows where the data is anomalous 
    df = df[df['trip_duration'].values <= trip_dur_max] #any rows where the trip is over 10 hours (3600 seconds)
    df = df[df['trip_duration'].values >= trip_dur_min] #at least 5 min
    df = df[df['distance'].values >= dist_min] #at least 0.5 km or 6 blocks

    return df
