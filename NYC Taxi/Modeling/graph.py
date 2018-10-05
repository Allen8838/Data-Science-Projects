import matplotlib.pyplot as plt
import numpy as np


def graph_train_test_maps(train, test, lat, long): 
    city_long_border = (-74.03, -73.75)
    city_lat_border = (40.63, 40.85)
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    ax[0].scatter(train[long].values[:100000], train[lat].values[:100000],
                color='blue', s=1, label='train', alpha=0.1)
    ax[1].scatter(test[long].values[:100000], test[lat].values[:100000],
                color='green', s=1, label='test', alpha=0.1)
    fig.suptitle('Train and test area complete overlap.')
    ax[0].legend(loc=0)
    ax[0].set_ylabel('latitude')
    ax[0].set_xlabel('longitude')
    ax[1].set_xlabel('longitude')
    ax[1].legend(loc=0)
    plt.ylim(city_lat_border)
    plt.xlim(city_long_border)
    plt.savefig('Train Test Maps.png')

    return None


def graph_train_test_trips(train, test, column):
    plt.plot(train.groupby(column).count()[['id']], 'o-', label='train')
    plt.plot(test.groupby(column).count()[['id']], 'o-', label='test')
    plt.title('Trips over Time')
    plt.legend(loc=0)
    plt.ylabel('Trips')
    plt.savefig('Train vs Test Trips Over time.png')

    return None

def graph_trip_dist(df, column):
    df['log_trip_duration'] = np.log(df['trip_duration'].values + 1)
    plt.hist(df['log_trip_duration'].values, bins=100)
    plt.xlabel('log(trip_duration)')
    plt.ylabel('number of train records')
    plt.savefig('Log Distribution of Trip Duration.png')