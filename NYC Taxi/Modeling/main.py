"""main module to run submodules"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from graph import graph_train_test_maps
from feature_importance import find_feature_imp
from data_preprocessing import remove_outliers
from feature_engineering import create_cols_distances,\
                                create_avg_speed_cols,\
                                modify_datetime_train,\
                                modify_datetime_test,\
                                find_center_points
from check import check_valid_test_dist, check_train_test_dist
from kmeans import find_kmeans_clusters_graph


if __name__ == "__main__":
    TRAIN = pd.read_csv('../train.csv')
    TEST = pd.read_csv('../test.csv')

    #log transform trip duration. we can then use the rmse scoring on the log values
    #to get rmsle
    TRAIN['log_trip_duration'] = np.log(TRAIN['trip_duration'].values + 1)

    """Create additional features"""
    TRAIN = create_cols_distances(TRAIN)
    TEST = create_cols_distances(TEST)

    TRAIN = find_center_points(TRAIN, 'pickup_latitude', 'pickup_longitude',
                               'dropoff_latitude', 'dropoff_longitude')

    TEST = find_center_points(TEST, 'pickup_latitude', 'pickup_longitude',
                              'dropoff_latitude', 'dropoff_longitude')

    #one hot encode the flag column first
    TRAIN = pd.get_dummies(TRAIN, columns=["store_and_fwd_flag"])
    TEST = pd.get_dummies(TEST, columns=["store_and_fwd_flag"])

    TRAIN = remove_outliers(TRAIN, 3600, 300, 0.5)

    #defining target here as this should now have the same number of rows
    #as the training set
    TARGET = TRAIN['log_trip_duration']

    TRAIN = create_avg_speed_cols(TRAIN)

    TRAIN = modify_datetime_train(TRAIN)
    TEST = modify_datetime_test(TEST)

    #reshape coordinates
    coords_train = np.vstack((TRAIN[['pickup_latitude', 'pickup_longitude']].values,
                              TRAIN[['dropoff_latitude', 'dropoff_longitude']].values))

    coords_test = np.vstack((TEST[['pickup_latitude', 'pickup_longitude']].values,
                             TEST[['dropoff_latitude', 'dropoff_longitude']].values))

    """Create cluster features"""
    find_kmeans_clusters_graph(TRAIN, coords_train,
                               'pickup_latitude', 'pickup_longitude',
                               'dropoff_latitude', 'dropoff_longitude',
                               'kmeans_clusters_train.png')

    find_kmeans_clusters_graph(TEST, coords_test,
                               'pickup_latitude', 'pickup_longitude',
                               'dropoff_latitude', 'dropoff_longitude',
                               'kmeans_clusters_test.png')

    features_not_used = ['id', 'log_trip_duration',
                         'pickup_datetime', 'dropoff_datetime',
                         'dropoff_datetime', 'trip_duration',
                         'pickup_date', 'dropoff_date',
                         'avg_speed_haversine', 'avg_speed_manhattan',
                         'dropoff_hour', 'dropoff_minute',
                         'dropoff_hour_sin', 'dropoff_hour_cos',
                         'dropoff_day_0', 'dropoff_day_1',
                         'dropoff_day_2', 'dropoff_day_3',
                         'dropoff_day_4', 'dropoff_day_5',
                         'dropoff_day_6', 'pickup_day_0',
                         'pickup_day_1', 'pickup_day_2',
                         'pickup_day_3', 'pickup_day_4',
                         'pickup_day_5', 'pickup_day_6']

    features_used = [f for f in TRAIN.columns if f not in features_not_used]

    """Sanity check to make sure that the distribuutions look right before modeling"""
    check_train_test_dist(TRAIN, TEST, features_used)
    graph_train_test_maps(TRAIN, TEST, 'pickup_latitude', 'pickup_longitude')

    Xtr, Xv, ytr, yv = train_test_split(TRAIN[features_used].values, TARGET,
                                        test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xv, label=yv)
    dtest = xgb.DMatrix(TEST[features_used].values)
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

    check_valid_test_dist(model, dtest, TEST, dvalid)
    
    