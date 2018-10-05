import pandas as pd 
import matplotlib.pyplot as plt
import datetime as dt
from IPython.core.pylabtools import figsize
import numpy as np
import seaborn as sns
import xgboost as xgb
from graph import graph_train_test_maps, graph_train_test_trips
from feature_importance import find_feature_imp
from data_preprocessing import get_desc_stats,\
                               missing_values_table,\
                               haversine_array,\
                               dummy_manhattan_distance,\
                               bearing_array,\
                               convert_time_sin_cos,\
                               modify_datetime
from check import check_valid_test_dist
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
    t0 = dt.datetime.now()    
    train = pd.read_csv('../train.csv')
    test = pd.read_csv('../test.csv')

    #log transform trip duration. we can then use the rmse scoring on the log values
    #to get rmsle
    train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)

    train = create_cols_distances(train)
    test = create_cols_distances(test)

    #one hot encode the flag column first
    train = pd.get_dummies(train, columns=["store_and_fwd_flag"])
    test = pd.get_dummies(test, columns=["store_and_fwd_flag"])

    train = remove_outliers(train, 3600, 300, 0.5)

    train = create_avg_speed_cols(train)

    train = modify_datetime(train)
    test = modify_datetime(test)

    # train.head(10000).to_csv('train10000.csv')
    # test.head(10000).to_csv('test10000.csv')
    
    coords_train = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                              train[['dropoff_latitude', 'dropoff_longitude']].values))

    coords_test = np.vstack((test[['pickup_latitude', 'pickup_longitude']].values,
                             test[['dropoff_latitude', 'dropoff_longitude']].values))


    find_kmeans_clusters_graph(train, coords_train, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'kmeans_clusters_train.png')
    find_kmeans_clusters_graph(test, coords_test, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'kmeans_clusters_test.png')
    
    features_not_used = ['id', 'log_trip_duration', 
                         'pickup_datetime', 'dropoff_datetime', 
                         'dropoff_datetime', 'trip_duration', 
                         'pickup_date', 'dropoff_date', 
                         'avg_speed_haversine','avg_speed_manhattan',
                         'dropoff_hour', 'dropoff_minute',
                         'dropoff_hour_sin', 'dropoff_hour_cos',
                         'dropoff_day_0','dropoff_day_1',
                         'dropoff_day_2', 'dropoff_day_3',
                         'dropoff_day_4', 'dropoff_day_5',
                         'dropoff_day_6', 'pickup_day_0', 
                         'pickup_day_1', 'pickup_day_2', 
                         'pickup_day_3','pickup_day_4',
                         'pickup_day_5','pickup_day_6']

    features_used = [f for f in train.columns if f not in features_not_used]
    target = np.log(train['trip_duration'].values + 1)

    # train = train.astype(float) 
    # test = test.astype(float)

    # train[features_used].head(1000).to_csv('subset of training.csv')
    # test.head(1000).to_csv('subset of test.csv')
    Xtr, Xv, ytr, yv = train_test_split(train[features_used].values, target, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xv, label=yv)
    dtest = xgb.DMatrix(test[features_used].values)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    xgb_pars = {'min_child_weight': 50, 'eta': 0.3, 'colsample_bytree': 0.3, 'max_depth': 10,
            'subsample': 0.8, 'lambda': 1., 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}

    model = xgb.train(xgb_pars, dtrain, 60, watchlist, early_stopping_rounds=50,
                  maximize=False, verbose_eval=10)

    print('Modeling RMSLE %.5f' % model.best_score)
    t1 = dt.datetime.now()
    print('Training time: %i seconds' % (t1 - t0).seconds)
    
    imp_features = find_feature_imp(model, features_used)

    imp_features.to_csv('feature_importances.csv')
     
    check_valid_test_dist(model, dtest, test, dvalid)
    

    
    