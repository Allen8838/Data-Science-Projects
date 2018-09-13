import pandas as pd
import datetime 

list_of_names_in_original_file = []

list_of_cleaned_names = []

adjust_for_milliseconds = 1000

add_to_list_names_original = lambda x: list_of_names_in_original_file.append(x) 
add_to_list_cleaned_names = lambda x: list_of_cleaned_names.append(x) 
to_lowercase = lambda x:x.lower()
#leaving out time so that the data is easier to visualize graphically. leaving out day also as too many unique days
#slows down matplotlib when plotting
convert_time_readable = lambda x: datetime.datetime.fromtimestamp(int(x/adjust_for_milliseconds)).strftime('%Y-%m')  

def place_names_in_original_and_cleaned_names_in_lists(df_dictionary_to_clean_names):
    df_dictionary_to_clean_names['name in original file'].apply(add_to_list_names_original)
    df_dictionary_to_clean_names['cleaned name in new file'].apply(add_to_list_cleaned_names)

    return list_of_names_in_original_file, list_of_cleaned_names


def replace_messy_names_w_cleaned_names(df, list_of_names_in_original_file, list_of_cleaned_names, specific_column=None):
    #apply cleaning to all columns of dataframe
    if specific_column == None:
        for column in df:
            df[column] = df[column].replace(list_of_names_in_original_file, list_of_cleaned_names)
    #apply cleaning to only one column of dataframe
    else:
        df[specific_column] = df[specific_column].replace(list_of_names_in_original_file, list_of_cleaned_names)

    return df 


def convert_names_to_lowercase(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].apply(to_lowercase)

    return dataframe


def convert_unix_time_to_utc_time(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].map(convert_time_readable)

    return dataframe










