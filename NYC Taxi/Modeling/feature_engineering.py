"""create additional features"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_cols_distances(df):
    #create a column for haversine distance
    df['distance'] = haversine_array(df['pickup_longitude'], df['pickup_latitude'],
                                     df['dropoff_longitude'], df['dropoff_latitude'])

    df['manhattan_distance'] = dummy_manhattan_distance(df['pickup_longitude'], df['pickup_latitude'],
                                                        df['dropoff_longitude'], df['dropoff_latitude'])

    df['bearing'] = bearing_array(df['pickup_longitude'], df['pickup_latitude'],
                                  df['dropoff_longitude'], df['dropoff_latitude'])

    return df


def create_avg_speed_cols(df):
    #create speed column. this should be correlated with the day component
    #and may give additional insight
    df['avg_speed_haversine'] = 1000*df['distance'].values/df['trip_duration']
    df['avg_speed_manhattan'] = 1000*df['manhattan_distance'].values/df['trip_duration']

    return df 


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
    """calculates direction the taxi is traveling"""
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

def find_center_points(df, lat1, long1, lat2, long2):
    df['center_latitude'] = (df[lat1].values + df[long2].values) / 2
    df['center_longitude'] = (df[long1].values + df[lat2].values) / 2

    return df


def modify_datetime(df):
    df['pickup_hour'] = pd.to_datetime(df['pickup_datetime']).dt.hour

    #test set does not have dropoff_datetime
    try: 
        df['dropoff_hour'] = pd.to_datetime(df['dropoff_datetime']).dt.hour
    except:
        pass

    df['pickup_minute'] = pd.to_datetime(df['pickup_datetime']).dt.minute 

    try:
        df['dropoff_minute'] = pd.to_datetime(df['dropoff_datetime']).dt.minute
    except:
        pass

    df['pickup_hour_sin'], df['pickup_hour_cos'] = convert_time_sin_cos(df, 'pickup_hour')

    try:
        df['dropoff_hour_sin'], df['dropoff_hour_cos'] = convert_time_sin_cos(df, 'dropoff_hour')
    except:
        pass
    #split datetime between dates and time
    #using normalize even though it gives us 0:00 time, but the resulting column is a datetime object, which allows us to further process
    #for day of week
    df['pickup_date'] = pd.to_datetime(df['pickup_datetime']).dt.date

    try:
        df['dropoff_date'] = pd.to_datetime(df['dropoff_datetime']).dt.date
    except:
        pass
     
    #create day of the week for both pickup date and dropoff dates
    df['pickup_day'] = pd.to_datetime(df['pickup_datetime']).dt.weekday

    try:
        df['dropoff_day'] = pd.to_datetime(df['dropoff_datetime']).dt.weekday
    except:
        pass

    #get week of year to capture effects of holidays 
    df['pickup_weekofyear'] = pd.to_datetime(df['pickup_datetime']).dt.weekofyear

    df["month"] = pd.to_datetime(df['pickup_datetime']).dt.month

    df["year"] = pd.to_datetime(df['pickup_datetime']).dt.year
    #one hot encode day of the week for both pickup and dropoff
    try:
        df = pd.get_dummies(df, columns=['pickup_day', 'dropoff_day'])
    except:
        pass

    return df 