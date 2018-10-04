import pandas as pd 
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import numpy as np
import seaborn as sns
import xgboost as xgb
from graph import graph_train_test_maps, graph_train_test_trips
from data_preprocessing import get_desc_stats,\
                               missing_values_table,\
                               haversine_array,\
                               dummy_manhattan_distance,\
                               bearing_array,\
                               convert_time_sin_cos,\
                               modify_datetime

from kmeans import find_kmeans_clusters_graph
from sklearn.model_selection import train_test_split


def create_cols_distances(df):
    #create a column for haversine distance
    df['distance'] = haversine_array(df['pickup_longitude'], df['pickup_latitude'], df['dropoff_longitude'], df['dropoff_latitude'])

    df['manhattan_distance'] = dummy_manhattan_distance(df['pickup_longitude'], df['pickup_latitude'], df['dropoff_longitude'], df['dropoff_latitude'])

    df['bearing'] = bearing_array(df['pickup_longitude'], df['pickup_latitude'], df['dropoff_longitude'], df['dropoff_latitude'])

    return df

def remove_outliers(df, trip_dur_max, trip_dur_min, dist_min):
    #delete select rows where the data is anomalous 
    df = df[df['trip_duration'].values <= trip_dur_max] #any rows where the trip is over 10 hours (3600 seconds)
    df = df[df['trip_duration'].values >= trip_dur_min] #at least 5 min
    df = df[df['distance'].values >= dist_min] #at least 0.5 km or 6 blocks

    return df

def create_avg_speed_cols(df):
    #create speed column. this should be correlated with the day component
    #and may give additional insight
    df['avg_speed_haversine'] = 1000*df['distance'].values/df['trip_duration']
    df['avg_speed_manhattan'] = 1000*df['manhattan_distance'].values/df['trip_duration']

    return df 


if __name__ == "__main__":    
    train = pd.read_csv('../train.csv')
    test = pd.read_csv('../test.csv')

    train = create_cols_distances(train)
    test = create_cols_distances(test)

    #one hot encode the flag column first
    train = pd.get_dummies(train, columns=["store_and_fwd_flag"])
    test = pd.get_dummies(test, columns=["store_and_fwd_flag"])

    train = remove_outliers(train, 3600, 300, 0.5)

    train = create_avg_speed_cols(train)
    test = create_avg_speed_cols(test)

    train = modify_datetime(train)
    test = modify_datetime(test)
    
    coords_train = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                              train[['dropoff_latitude', 'dropoff_longitude']].values))

    coords_test = np.vstack((test[['pickup_latitude', 'pickup_longitude']].values,
                             test[['dropoff_latitude', 'dropoff_longitude']].values))


    find_kmeans_clusters_graph(train, coords_train, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'kmeans_clusters_train.png')
    find_kmeans_clusters_graph(test, coords_test, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'kmeans_clusters_test.png')

    

    

    

     

 

    
    