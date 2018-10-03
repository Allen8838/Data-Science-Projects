import pandas as pd 
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import numpy as np
import seaborn as sns
from data_preprocessing import get_desc_stats, ,missing_values_table, haversine_array, dummy_manhattan_distance, bearing_array, convert_datetime_n_round, split_date_time, find_day_of_week

def create_cols_distances():
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

def create_avg_speed_cols():
    #create speed column. this should be correlated with the day component
    #and may give additional insight
    df['avg_speed_haversine'] = 1000*df['distance'].values/df['trip_duration']
    df['avg_speed_manhattan'] = 1000*df['manhattan_distance'].values/df['trip_duration']

    return df 



if __name__ == "__main__":    
    df = pd.read_csv('../train.csv')    
    get_desc_stats(df, 'descriptive statistics.csv', 'histograms_of_columns.png')
    #see which columns has missing values and look at how much missing values are there
    missing_values_table(df)

    df = create_cols_distances()

    #one hot encode the flag column first
    df = pd.get_dummies(df, columns=["store_and_fwd_flag"])

    df = remove_outliers(df, 3600, 300, 0.5)

    df = create_avg_speed_cols()

    #convert pickup_date and dropoff_date to datetime objects
    #round both pickup and dropoff time to the latest 15 min before applying sin and cos functions to them
    #assuming that hours will matter, not minutes and want to get to the closest hour
    
    df['pickup_datetime_rounded_15_min'] = convert_datetime_n_round(df, 'pickup_datetime', '15min')
    df['dropoff_datetime_rounded_15_min'] = convert_datetime_n_round(df, 'dropoff_datetime', '15min')

    df['rounded_pickup_time_hour'] = pd.to_datetime(df['pickup_datetime_rounded_15_min']).dt.hour 
    df['rounded_dropoff_time_hour'] = pd.to_datetime(df['dropoff_datetime_rounded_15_min']).dt.hour 

    df['pickup_time_hour_sin'] = np.sin(2 * np.pi * df['rounded_pickup_time_hour']/23.0)
    df['pickup_time_hour_cos'] = np.cos(2 * np.pi * df['rounded_pickup_time_hour']/23.0)

    df['dropoff_time_hour'] = np.sin(2 * np.pi * df['rounded_dropoff_time_hour']/23.0)
    df['rounded_dropoff_time_hour'] = np.cos(2 * np.pi * df['rounded_dropoff_time_hour']/23.0)

    #split datetime between dates and time
    #using normalize even though it gives us 0:00 time, but the resulting column is a datetime object, which allows us to further process
    #for day of week
    df['pickup_date'], df['pickup_time'] = split_date_time(df, 'pickup_datetime_rounded_15_min')
    df['dropoff_date'], df['dropoff_time'] = split_date_time(df, 'dropoff_datetime_rounded_15_min')
     
    #create day of the week for both pickup date and dropoff dates
    df['pickup_day_of_week'] = find_day_of_week(df, 'pickup_datetime_rounded_15_min') 

    df['dropoff_day_of_week'] = find_day_of_week(df, 'dropoff_datetime_rounded_15_min')

    #one hot encode day of the week for both pickup and dropoff
    df = pd.get_dummies(df, columns=['pickup_day_of_week', 'dropoff_day_of_week'])

    df.head(10000).to_csv('dataframe_w_datetime_feature_engineering.csv')
    correlations_data = df.corr()['trip_duration'].sort_values()

    correlations_data.to_csv("correlation_data.csv")



    #shorten the dataframe by whether the trip duration column is within 3 standard deviations
    #there are extreme trip duration values that severely distorts the histogram
    #df = df[np.abs(df.trip_duration-df.trip_duration.mean()) <= (3*df.trip_duration.std())]
    

    # # Histogram of the Trip Duration
    #figsize(8, 8)
    
    # plt.hist(df['trip_duration'].dropna(), bins = 100, edgecolor = 'k')
    # plt.xlabel('trip_duration') 
    # plt.ylabel('Number of Trips')
    # plt.title('Trip Duration Distribution')
    # plt.tight_layout()
    # plt.savefig('histograms_of_trip_duration_within_3_std.png')

    #let's look at trip distribution based on vendors
    # df_vendor_1 = df[df['vendor_id'] == 1]
    # df_vendor_2 = df[df['vendor_id'] == 2]


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

    #break histogram into buckets as we have some extreme values, distorting the histogram to look like one column
    


    # # Plot of distribution of scores for passenger categories
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

    """Find all correlations with trip_duration and sort"""
    """Feature Engineering with Dates"""
    

 

    print(df.shape)
    #feature engineering with datetime objects
    #split data from time for pickup and dropoff datetime object
    

    #create pair plot 
    #data_pairplot = sns.pairplot(df)
    #data_pairplot.savefig('pairplot.png')
