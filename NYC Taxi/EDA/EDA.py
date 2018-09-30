import pandas as pd 
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import numpy as np


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
    df = df[np.abs(df.trip_duration-df.trip_duration.mean()) <= (3*df.trip_duration.std())]
    

    # # Histogram of the Trip Duration
    figsize(8, 8)
    
    plt.hist(df['trip_duration'].dropna(), bins = 1000, edgecolor = 'k')
    plt.xlabel('trip_duration') 
    plt.ylabel('Number of Trips')
    plt.title('Trip Duration Distribution')
    plt.tight_layout()
    plt.savefig('histograms_of_trip_duration_within_3_std.png')

    #look at datatype of each column
    #df.info().to_csv('datatype_desc.csv')

    #get a small subset of the data for viewing
    #df.head(10000).to_csv('first_10000_dataset_records.csv')

    #break histogram into buckets as we have some extreme values, distorting the histogram to look like one column
    