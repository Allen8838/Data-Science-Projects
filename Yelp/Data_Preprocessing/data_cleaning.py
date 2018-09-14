import json
import pandas as pd 
import csv 
import math
import numpy as np 

remove_b_tag = lambda x: x[1:] if type(x)==str and x[0] =='b' else (x if type(x)==float else x)

def remove_b_tag_from_csv(csv_file):
    df = pd.read_csv(csv_file)

    for column in df:
        df[column] = df[column].map(remove_b_tag)

    return df
    
def write_dataframe_to_csv_file(filename, dataframe):
    dataframe.to_csv(filename, encoding='utf-8', index=False)

    return None

def filter_business_by_category(dataframe, category):
    #remove any rows that are blank as they will cause a problem when we use the apply function below
    dataframe_wo_NAs = dataframe.dropna(subset=['categories'])
    
    df_filtered = dataframe_wo_NAs[dataframe_wo_NAs['categories'].apply(lambda x: True if category in x else False)]
    
    return df_filtered

def select_certain_columns_in_dataframe(dataframe, list_of_columns):
    dataframe_w_certain_columns = dataframe[list_of_columns]

    return dataframe_w_certain_columns

def rename_columns_in_dataframe(dataframe, original, new):
    dataframe.rename(columns={original:new}, inplace=True)

    return dataframe


if __name__ == '__main__':
    ###Collecting restaurant business data###

    #write_dataframe_to_csv_file('yelp_academic_dataset_user.csv', remove_b_tag_from_csv('../Yelp Dataset/csv/yelp_academic_dataset_user.csv'))
    df = pd.read_csv(r'../Yelp Dataset/csv/yelp_academic_dataset_business.csv')
    df_filtered = filter_business_by_category(df, 'Restaurants')
    
    #placing business_id first as we will want to use this index to join the review file
    selected_columns = ['business_id','attributes.RestaurantsPriceRange2', 'attributes.RestaurantsGoodForGroups', 'city', 
    'categories', 'attributes.GoodForMeal', 'attributes.RestaurantsAttire', 'attributes.HasTV', 'attributes.RestaurantsDelivery',
    'attributes.NoiseLevel', 'attributes.RestaurantsTakeOut', 'attributes.GoodForKids', 'attributes.Open24Hours', 'attributes.BusinessParking',
    'attributes.WiFi', 'state', 'attributes', 'is_open', 'attributes.Alcohol', 'stars', 'attributes.RestaurantsTableService',
    'name', 'attributes.Ambience']

    #the column for stars should really be avg_stars. renaming this column and reducing the number of columns in final dataframe
    dataframe_w_certain_columns = rename_columns_in_dataframe(select_certain_columns_in_dataframe(df_filtered, selected_columns), 'stars', 'avg_stars')
    
    #get the unique business_id and then later collect only the reviews with these ids
    #every id should be unique as this is the business file and each line is about one business
    list_unique_business_id = dataframe_w_certain_columns.business_id.unique()

    write_dataframe_to_csv_file('business_file_restaurants_certain_columns.csv', dataframe_w_certain_columns)

    

    