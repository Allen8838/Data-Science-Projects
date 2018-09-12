import pandas as pd
import datetime 

list_of_names_in_original_file = []

list_of_cleaned_names = []

adjust_for_milliseconds = 1000

add_to_list_names_original = lambda x: list_of_names_in_original_file.append(x) 
add_to_list_cleaned_names = lambda x: list_of_cleaned_names.append(x) 
to_lowercase = lambda x:x.lower()
convert_time_readable = lambda x: datetime.datetime.fromtimestamp(int(x/adjust_for_milliseconds)).strftime('%Y-%m-%d')  #leaving out time so that the data is easier to visualize graphically


def place_names_in_original_and_cleaned_names_in_lists(df_dictionary_to_clean_names):
    df_dictionary_to_clean_names['name in original file'].apply(add_to_list_names_original)
    df_dictionary_to_clean_names['cleaned name in new file'].apply(add_to_list_cleaned_names)

    return list_of_names_in_original_file, list_of_cleaned_names


def replace_messy_names_w_cleaned_names(df, list_of_names_in_original_file, list_of_cleaned_names):
    df['sender'] = df['sender'].replace(list_of_names_in_original_file, list_of_cleaned_names)

    return df 


def convert_names_to_lowercase(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].apply(to_lowercase)

    return dataframe


def convert_unix_time_to_utc_time(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].map(convert_time_readable)

    return dataframe










