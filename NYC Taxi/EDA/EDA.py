import pandas as pd 
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import numpy as np
import seaborn as sns


df = pd.read_csv('../train.csv')


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





if __name__ == "__main__":
    # print(list(df.columns.values))
    # df.describe().to_csv('descriptive statistics.csv')
    # df.hist()
    # plt.tight_layout()
    # plt.savefig('histograms_of_columns.png')
    #print(df.T.apply(lambda x: x.nunique(), axis=1))
    
    #see which columns has missing values and look at how much missing values are there
    #missing_values_table(df)

    #delete the row value with the outlier trip duration as it is skewing the trip duration histogram dramatically
    #row_to_delete = df['trip_duration'].idxmax()
    #df.drop(df.index[row_to_delete])

    #print(df.nlargest(10, 'trip_duration'))

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



    #look at datatype of each column
    #df.info().to_csv('datatype_desc.csv')

    #get a small subset of the data for viewing
    #df.head(10000).to_csv('first_10000_dataset_records.csv')

    #break histogram into buckets as we have some extreme values, distorting the histogram to look like one column
    

    #density plot on passengar count. may tell us if passengar count can predict trip duration
    #print(df['passenger_count'].nunique()) 10 unique passenger_count values

    #print(df['passenger_count'].unique())
    
    # df = df[np.abs(df.trip_duration-df.trip_duration.mean()) <= (3*df.trip_duration.std())]
    # list_passenger_count_unique = df['passenger_count'].unique()


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
    
    
    # #one hot encode the flag column first
    df = pd.get_dummies(df, columns=["store_and_fwd_flag"])
    #feature engineering with datetime objects
    #split data from time for pickup and dropoff datetime object
    
    #convert pickup_date and dropoff_date to datetime objects
    df['pickup_datetime'] =  pd.to_datetime(df['pickup_datetime'])
    #using normalize even though it gives us 0:00 time, but the resulting column is a datetime object, which allows us to further process
    #for day of week
    df['pickup_date'] = df['pickup_datetime'].dt.normalize()
    df['pickup_time'] = df['pickup_datetime'].dt.time

    df['dropoff_datetime'] =  pd.to_datetime(df['dropoff_datetime'])
    df['dropoff_date'] = df['dropoff_datetime'].dt.normalize()
    df['dropoff_time'] = df['dropoff_datetime'].dt.time
    
    #create day of the week for both pickup date and dropoff dates
    df['pickup_day_of_week'] = df['pickup_date'].dt.day_name()

    df['dropoff_day_of_week'] = df['dropoff_date'].dt.day_name()

    #one hot encode day of the week for both pickup and dropoff
    df = pd.get_dummies(df, columns=['pickup_day_of_week', 'dropoff_day_of_week'])
    
    print(df.head())
    # correlations_data = df.corr()['trip_duration'].sort_values()

    # correlations_data.to_csv("correlation_data.csv")

