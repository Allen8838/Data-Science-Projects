"""Replace names in original data file with self built name mapping csv file,
convert all names to the same case,
convert unix time to regular date time"""

import datetime

def place_orig_clean_names_to_ls(df_clean_names):
    """move names from the df into lists to make it easier to replace names"""

    orig_names = df_clean_names['name in original file'].tolist()
    clean_names = df_clean_names['cleaned name in new file'].tolist()

    return orig_names, clean_names


def replace_messy_w_clean_names(df, orig_names, clean_names, specific_column=None):
    """assumes that the df has only one column by default.
    if df has multiple columns, specify the column to change
    """
    if specific_column is None:
        for column in df:
            df[column] = df[column].replace(orig_names, clean_names)
    #apply cleaning to only one column of df
    else:
        df[specific_column] = df[specific_column].replace(orig_names, clean_names)

    return df


def convert_names_to_lowercase(df, column_name):
    """this makes sure that the names are in one format so that they
    can be grouped and counted together in later analysis"""

    #use numpy vectorization 
    df[column_name] = df[column_name].str.lower()

    return df


def convert_unix_time_to_utc_time(df, column_name):
    """replaceing unix time with human readable date for graphing"""
    adj_for_ms = 1000
    #leaving out time so that the data is easier to visualize graphically. leaving out day
    # also as too many unique days slows down matplotlib when plotting
    convert_time_readable = lambda x: datetime.datetime.fromtimestamp(int(x/adj_for_ms)).strftime('%Y-%m')

    df[column_name] = df[column_name].map(convert_time_readable)

    return df
