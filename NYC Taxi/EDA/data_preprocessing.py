import pandas as pd 
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import numpy as np
import seaborn as sns

def get_desc_stats(df, csv_name, png_name):
    df.describe().to_csv(csv_name)
    df.hist()
    plt.tight_layout()
    plt.savefig(png_name)

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

def haversine_array(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    R = 6371.0  # radius of the earth in km

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = R * c
    return km

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


def convert_time_sin_cos(df, column):
    sin_var = np.sin(2 * np.pi * df[column]/23.0)
    cos_var = np.cos(2 * np.pi * df[column]/23.0)

    return sin_var, cos_var


def modify_datetime(df):
    df['pickup_hour'] = pd.to_datetime(df['pickup_datetime']).dt.hour 
    df['dropoff_hour'] = pd.to_datetime(df['dropoff_datetime']).dt.hour

    df['pickup_minute'] = pd.to_datetime(df['pickup_datetime']).dt.minute 
    df['dropoff_minute'] = pd.to_datetime(df['dropoff_datetime']).dt.minute

    df['pickup_hour_sin'], df['pickup_hour_cos'] = convert_time_sin_cos(df, 'pickup_hour')
    df['dropoff_hour_sin'], df['dropoff_hour_cos'] = convert_time_sin_cos(df, 'dropoff_hour')
     
    #split datetime between dates and time
    #using normalize even though it gives us 0:00 time, but the resulting column is a datetime object, which allows us to further process
    #for day of week
    df['pickup_date'] = pd.to_datetime(df['pickup_datetime']).dt.date
    df['dropoff_date'] = pd.to_datetime(df['dropoff_datetime']).dt.date
     
    #create day of the week for both pickup date and dropoff dates
    df['pickup_day'] = pd.to_datetime(df['pickup_datetime']).dt.weekday

    df['dropoff_day'] = pd.to_datetime(df['dropoff_datetime']).dt.weekday

    #get week of year to capture effects of holidays 
    df['pickup_weekofyear'] = pd.to_datetime(df['pickup_datetime']).dt.weekofyear

    df["month"] = pd.to_datetime(df['pickup_datetime']).dt.month

    df["year"] = pd.to_datetime(df['pickup_datetime']).dt.year
    #one hot encode day of the week for both pickup and dropoff
    df = pd.get_dummies(df, columns=['pickup_day', 'dropoff_day'])

    return df 