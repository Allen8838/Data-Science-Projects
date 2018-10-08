"""sanity checks to make sure that training and testing distribution matches"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def check_valid_test_dist(model, dtest, test, dvalid):
    ypred = model.predict(dvalid)
    ytest = model.predict(dtest)

    print('Test shape OK.') if test.shape[0] == ytest.shape[0] else print('Oops')
    test['trip_duration'] = np.exp(ytest) - 1
    test[['id', 'trip_duration']].to_csv('allen_xgb_submission.csv.gz', index=False, compression='gzip')

    print('Valid prediction mean: %.3f' % ypred.mean())
    print('Test prediction mean: %.3f' % ytest.mean())

    _, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    sns.distplot(ypred, ax=ax[0], color='blue', label='validation prediction')
    sns.distplot(ytest, ax=ax[1], color='green', label='test prediction')
    ax[0].legend(loc=0)
    ax[1].legend(loc=0)
    plt.savefig('Validation and Test Distributions.png')
    

def check_train_test_dist(train, test, features_used):
    feature_stats = pd.DataFrame({'feature': features_used})
    feature_stats.loc[:, 'train_mean'] = np.nanmean(train[features_used].values, axis=0).round(4)
    feature_stats.loc[:, 'test_mean'] = np.nanmean(test[features_used].values, axis=0).round(4)

    feature_stats.loc[:, 'train_std'] = np.nanstd(train[features_used].values, axis=0).round(4)
    feature_stats.loc[:, 'test_std'] = np.nanstd(test[features_used].values, axis=0).round(4)

    feature_stats.loc[:, 'train_nan'] = np.mean(np.isnan(train[features_used].values), axis=0).round(3)
    feature_stats.loc[:, 'test_nan'] = np.mean(np.isnan(test[features_used].values), axis=0).round(3)

    feature_stats.loc[:, 'train_test_mean_diff'] = np.abs(feature_stats['train_mean'] - feature_stats['test_mean']) / np.abs(feature_stats['train_std'] + feature_stats['test_std'])  * 2
    feature_stats.loc[:, 'train_test_nan_diff'] = np.abs(feature_stats['train_nan'] - feature_stats['test_nan'])
    feature_stats = feature_stats.sort_values(by='train_test_mean_diff')
    feature_stats[['feature', 'train_test_mean_diff']].tail()
    
    feature_stats = feature_stats.sort_values(by='train_test_nan_diff')
    feature_stats[['feature', 'train_nan', 'test_nan', 'train_test_nan_diff']].tail()

    feature_stats.to_csv('Check Train and Test Distributions.csv')

    return None
