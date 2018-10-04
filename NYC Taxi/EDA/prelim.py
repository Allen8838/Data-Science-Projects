
    get_desc_stats(df, 'descriptive statistics.csv', 'histograms_of_columns.png')
    #see which columns has missing values and look at how much missing values are there
    missing_values_table(df)


    df.head(10000).to_csv('dataframe_w_datetime_feature_engineering.csv')
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
   