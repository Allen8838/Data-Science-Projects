import pandas as pd 
import matplotlib.pyplot as plt
import datetime as dt
from IPython.core.pylabtools import figsize
import numpy as np
import seaborn as sns
import xgboost as xgb
from graph import graph_train_test_maps, graph_train_test_trips, graph_trip_dist
from feature_importance import find_feature_imp
from data_preprocessing import missing_values_table
from check import check_valid_test_dist, check_train_test_dist
from kmeans import find_kmeans_clusters_graph
from sklearn.model_selection import train_test_split


def get_desc_stats(df, csv_name, png_name):
    df.describe().to_csv(csv_name)
    df.hist()
    plt.tight_layout()
    plt.savefig(png_name)


if __name__ == '__main__':
    """EDA 1a - Check Distribution of Trip Duration"""
    train = pd.read_csv('../train.csv')
    #shorten the dataframe by whether the trip duration column is within 3 standard deviations
    #there are extreme trip duration values that severely distorts the histogram
    #train = train[np.abs(train.trip_duration-train.trip_duration.mean()) <= (3*df.trip_duration.std())]
    

    # # Histogram of the Trip Duration
    #figsize(8, 8)
    
    # plt.hist(train['trip_duration'].dropna(), bins = 100, edgecolor = 'k')
    # plt.xlabel('trip_duration') 
    # plt.ylabel('Number of Trips')
    # plt.title('Trip Duration Distribution')
    # plt.tight_layout()
    # plt.savefig('histograms_of_trip_duration_within_3_std.png')

    """EDA 1b - Check Distribution of Trip Duration based on Vendors"""
    # df_vendor_1 = train[train['vendor_id'] == 1]
    # df_vendor_2 = train[train['vendor_id'] == 2]

    # plt.hist(df_vendor_1['trip_duration'].dropna(), bins = 100, edgecolor = 'k')
    # plt.xlabel('trip_duration') 
    # plt.ylabel('Number of Trips')
    # plt.title('Trip Duration Distribution for Vendor 1')
    # plt.tight_layout()
    # plt.savefig('Vendor_1_histograms_of_trip_duration_within_3_std.png')
    
    # plt.hist(df_vendor_2['trip_duration'].dropna(), bins = 100, edgecolor = 'k')
    # plt.xlabel('trip_duration') 
    # plt.ylabel('Number of Trips')
    # plt.title('Trip Duration Distribution for Vendor 2')
    # plt.tight_layout()
    # plt.savefig('Vendor_2_histograms_of_trip_duration_within_3_std.png')

    """EDA 1c - Check Log Distribution of Trip Duration"""
    
    train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)

    graph_trip_dist(train, 'trip_duration')

    """EDA 2 - Get some basic statistics of the columns and create histograms of columns"""
    get_desc_stats(df, 'descriptive statistics.csv', 'histograms_of_columns.png')
    
    """EDA 3 - See if any of the columns have missing values"""
    missing_values_table(df)

    """EDA 4 - See how each feature is correlated with one another. Save to CSV file"""
    correlations_data = df.corr()['trip_duration'].sort_values()

    correlations_data.to_csv("correlation_data.csv")

    """EDA 5 - Find pickup/dropoff clusters. Save the result to PNG"""

    coords = np.vstack((df[['pickup_latitude', 'pickup_longitude']].values,
                        df[['dropoff_latitude', 'dropoff_longitude']].values))

    find_kmeans_clusters_graph(df, coords,
                               'pickup_latitude', 'pickup_longitude',
                               'dropoff_latitude', 'dropoff_longitude',
                               'kmeans_clusters.png')

    """EDA 6 - See how the number of passengers change with the Vendor ID"""
    f, ax = plt.subplots(figsize=(20,5), ncols=1)
    pass_cnt_vendorid = sns.countplot("passenger_count", hue='vendor_id', data=df, ax=ax)
    figure = pass_cnt_vendorid.get_figure()
    _ = ax.set_xlim([0.5, 7])
    figure.savefig('Passenger_Count_vs_Vendor_ID.png')

    """EDA 7 - See if clusters differ based on Days""" 
    #assuming that the resulting cluster by days will be comparable to using
    #dropoff_days
    pickup_days = ['pickup_day_1', 
            'pickup_day_2',
            'pickup_day_3',
            'pickup_day_4',
            'pickup_day_5',
            'pickup_day_6',
            'pickup_day_0']

    output_img_names = ['KMeans_Monday.png', 
                        'KMeans_Tuesday.png',
                        'KMeans_Wednesday.png',
                        'KMeans_Thursday.png',
                        'KMeans_Friday.png',
                        'KMeans_Saturday.png',
                        'KMeans_Sunday.png']

    for i, day in enumerate(pickup_days):
        find_kmeans_clusters_graph(df[df[day]==1], coords,
                                   'pickup_latitude', 'pickup_longitude',
                                   'dropoff_latitude', 'dropoff_longitude', 
                                   output_img_names[i])
    

    """EDA 8 - See how features are correlated with each other"""
    #data_pairplot = sns.pairplot(df)
    #data_pairplot.savefig('pairplot.png')

    #graph_train_test_trips(train, test, 'pickup_date')
    #create pair plot     

    """EDA 9 - Plot of distribution of trip duration for passenger categories"""
    # figsize(12, 10)

    # # Plot each passenger count
    # for passenger_count in list_passenger_count_unique:
    #     if passenger_count == 0 or passenger_count == 1 or passenger_count == 2:
    #         # Select the passenger count type
    #         subset = df[df['passenger_count'] == passenger_count]
        
    #         # Density plot of passenger_count
    #         sns.kdeplot(subset['trip_duration'].dropna(),
    #                 label = passenger_count, shade = False, alpha = 0.8)
        
    # # label the plot
    # plt.xlabel('Trip Duration', size = 20); plt.ylabel('Density', size = 20)
    # plt.title('Density_Plot_of_Trip_Duration_by_PassCount-0-2', size = 28)
    # plt.savefig('Density_Plot_of_Trip_Duration_by_PassCount-0-2.png')

    