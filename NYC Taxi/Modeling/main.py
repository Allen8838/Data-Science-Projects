import pandas as pd 
import matplotlib.pyplot as plt
import datetime as dt
from IPython.core.pylabtools import figsize
import numpy as np
import seaborn as sns
import xgboost as xgb
from graph import graph_train_test_maps, graph_train_test_trips, graph_trip_dist
from feature_importance import find_feature_imp
from data_preprocessing import get_desc_stats, missing_values_table, remove_outliers
from feature_engineering import haversine_array,\
                                dummy_manhattan_distance,\
                                bearing_array,\
                                convert_time_sin_cos,\
                                modify_datetime,\
                                find_center_points

from check import check_valid_test_dist, check_train_test_dist
from kmeans import find_kmeans_clusters_graph
from sklearn.model_selection import train_test_split



if __name__ == "__main__":
    train = pd.read_csv('../train.csv')
    test = pd.read_csv('../test.csv')

    #log transform trip duration. we can then use the rmse scoring on the log values
    #to get rmsle
    train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)
    target = train['log_trip_duration']

    """Create additional features"""
    train = create_cols_distances(train)
    test = create_cols_distances(test)

    train = find_center_points(train, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
    test = find_center_points(test, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')

    #one hot encode the flag column first
    train = pd.get_dummies(train, columns=["store_and_fwd_flag"])
    test = pd.get_dummies(test, columns=["store_and_fwd_flag"])

    train = remove_outliers(train, 3600, 300, 0.5)

    train = create_avg_speed_cols(train)

    train = modify_datetime(train)
    test = modify_datetime(test)

    #reshape coordinates
    coords_train = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                              train[['dropoff_latitude', 'dropoff_longitude']].values))

    coords_test = np.vstack((test[['pickup_latitude', 'pickup_longitude']].values,
                             test[['dropoff_latitude', 'dropoff_longitude']].values))

    """Create cluster features"""
    find_kmeans_clusters_graph(train, coords_train, 
                               'pickup_latitude', 'pickup_longitude',
                               'dropoff_latitude', 'dropoff_longitude',
                               'kmeans_clusters_train.png')
    
    find_kmeans_clusters_graph(test, coords_test,
                               'pickup_latitude', 'pickup_longitude',
                               'dropoff_latitude', 'dropoff_longitude',
                               'kmeans_clusters_test.png')
    
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

    """Sanity check to make sure that the distribuutions look right before modeling"""
    check_train_test_dist(train, test)
    graph_train_test_maps(train, test, 'pickup_latitude', 'pickup_longitude')

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
    
    imp_features = find_feature_imp(model, features_used)

    #saved features to file
    imp_features.to_csv('feature_importances.csv')
     
    check_valid_test_dist(model, dtest, test, dvalid)
    

    
    