"""
Replace names in original data file with self built name mapping csv file,
convert all names to the same case,
convert unix time to regular date time
"""

import datetime
import pandas as pd
from create_tuples import create_tuple_senders_time

def get_preprocessed_input_data(csv_file):
    """
    clean data first
    """

    # read in the dataset given and add headers
    df = pd.read_csv(csv_file, names=['time',
                                      'message_id',
                                      'sender',
                                      'recipients',
                                      'topic',
                                      'mode'])

    # read in dataset created to clean the dataset given file
    df_to_clean_names = pd.read_csv(r'Dictionary-to-clean-names.csv')

    df, dict_orig_cleaned_names = get_clean_data(df, df_to_clean_names)

    senders_time = create_tuple_senders_time(df)

    return df, dict_orig_cleaned_names, senders_time


def get_clean_data(df, df_to_clean_names):
    """
    clean data by:
    - replacing different names but same person into one name
    - converting all names to lowercase
    - convert unix time to regular time
    """
    original_and_cleaned_names = {}

    for _, row in df_to_clean_names.iterrows():
        original_and_cleaned_names[row['name in original file']] = row['cleaned name in new file']

    df = replace_messy_w_clean_names(df, original_and_cleaned_names, 'sender')

    df = convert_names_to_lowercase(df, 'sender')

    # converting the time from milliseconds to date (without time). having just the date
    # will make it easier to graph the data for questions 2 and 3
    df = convert_unix_time_to_utc_time(df, 'time')

    return df, original_and_cleaned_names


def replace_messy_w_clean_names(df, dict_orig_cleaned_names, specific_column=None):
    """
    assumes that the df has only one column by default.
    if df has multiple columns, specify the column to change
    """
    if specific_column is None:
        for column in df:
            df[column] = df[column].map(dict_orig_cleaned_names).fillna(df[column])
    # apply cleaning to only one column of df
    else:
        df[specific_column] = df[specific_column].map(dict_orig_cleaned_names).fillna(df[specific_column])

    return df


def convert_names_to_lowercase(df, column_name):
    """
    this makes sure that the names are in one format so that they
    can be grouped and counted together in later analysis
    """

    # use numpy vectorization
    df[column_name] = df[column_name].str.lower()

    return df


def convert_unix_time_to_utc_time(df, column_name):
    """
    replaceing unix time with human readable date for graphing
    """

    adj_for_ms = 1000
    # leaving out time so that the data is easier to visualize graphically. leaving out day
    # also as too many unique days slows down matplotlib when plotting
    convert_time_readable = lambda x: datetime.datetime.fromtimestamp(int(x/adj_for_ms)).strftime('%Y-%m')

    df[column_name] = df[column_name].map(convert_time_readable)

    return df
