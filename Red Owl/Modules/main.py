"""main module to run submodules"""
import time 
from multiprocessing import Pool
from collections import OrderedDict
import pandas as pd

import numpy as np

from data_cleaning import place_orig_clean_names_to_ls, replace_messy_w_clean_names, convert_names_to_lowercase, convert_unix_time_to_utc_time
from senders import create_dict_senders_num_msgs
from recipients import cnt_msgs_recvd_by_each_recip, create_column_headers, coll_cols_rows_tsenders_recip, parse_recipients
from senders_recipients import find_top_senders, create_recip_sender_time
from plot import graph_top_senders, find_uni_num_msgs_uni_time_per, graph_tsenders_uni_msgs

def exec_q1():
    """clean data first, create file for question 1 and 
    create intermediary values for questions 2 and 3"""
    number_of_senders_to_look = 5

    #read in the dataset given and add headers
    df = pd.read_csv(r'enron-event-history-all.csv', names=['time',
                                                            'message_id',
                                                            'sender',
                                                            'recipients',
                                                            'topic',
                                                            'mode'])
    #read in dataset created to clean the dataset given file
    df_dict_to_clean_names = pd.read_csv(r'Dictionary-to-clean-names.csv')

    #some sender names are not in a consistent format. placing sender names
    #in original file as well as a cleanup name list into lists.
    orig_names, cleaned_names = place_orig_clean_names_to_ls(df_dict_to_clean_names)

    #used dict so that we can use map function to clean names
    dict_orig_cleaned_names = dict(zip(orig_names, cleaned_names))

    df = replace_messy_w_clean_names(df, dict_orig_cleaned_names, 'sender')

    #converting all names to the same case so that same names of different cases
    #will be grouped together
    df = convert_names_to_lowercase(df, 'sender')

    #converting the time from milliseconds to date (without time). having just the date
    #will make it easier to graph the data for questions 2 and 3
    df = convert_unix_time_to_utc_time(df, 'time')

    #senders_num_msgs_per_time will be used for question 2.
    #senders_time will be used for question 3
    num_msgs_sent_sender, senders_num_msgs_per_time, senders_time = create_dict_senders_num_msgs(df)

    #sort by value in descending order. this will then be placed directly onto the CSV file.
    # we will also know who are the top 5 senders from sorting this first
    num_msgs_sent_sender_desc = OrderedDict(sorted(num_msgs_sent_sender.items(),
                                                   key=lambda t: t[1],
                                                   reverse=True))

    parse_recip_df = parse_recipients(df, dict_orig_cleaned_names)

    msgs_rcvd_by_recip = cnt_msgs_recvd_by_each_recip(parse_recip_df)

    num_msgs_sent_n_rcvd_per = []
    #size of msgs_rcvd_by_recip is larger than size of num_msgs_sent_sender_desc.
    #need to union the keys of these two dictionaries the top 5 senders will be
    #in this dictionary, so need to append the list with this dictionary first
    for key in num_msgs_sent_sender_desc:
        #getting the union of the keys in both dictionaries, so will only append
        #if the key DOESN'T already exist in the list
        if key not in num_msgs_sent_n_rcvd_per:
            num_msgs_sent_n_rcvd_per.append(key)

    for key in msgs_rcvd_by_recip:
        if key not in num_msgs_sent_n_rcvd_per:
            num_msgs_sent_n_rcvd_per.append(key)

    #creates the file asked for Question 1 and returns a list of the top 5 senders for question 2
    top_five_senders = find_top_senders(num_msgs_sent_sender_desc,
                                        msgs_rcvd_by_recip,
                                        num_msgs_sent_n_rcvd_per,
                                        number_of_senders_to_look)

    #top_five_senders, senders_num_msgs_per_time for question 2
    #return senders_time and parse_recip_df for question 3
    return top_five_senders, senders_num_msgs_per_time, senders_time, parse_recip_df

#perform datacleaning of parse_recip_df. should speed up execution of
#question 1 as the memory generated from this helper function
#will be released upon exit of the function


def exec_q2(top_five_senders, senders_num_msgs_per_time):
    """create graph over time of the top senders and number of messages sent"""
    graph_top_senders(top_five_senders, senders_num_msgs_per_time)


def exec_q3(top_five_senders, senders_time, parse_recip_df):
    """perform additional processing of the expanded recipient dataframe columns.
    find top senders and the unique number of recipients they sent messages to and
    graph results"""
    mod_recip_df, col_n_rows_w_top_senders = coll_cols_rows_tsenders_recip(parse_recip_df,
                                                                           top_five_senders)

    recip_sender_time_sorted = create_recip_sender_time(senders_time,
                                                        mod_recip_df,
                                                        col_n_rows_w_top_senders,
                                                        top_five_senders)

    tsender_uni_num_msgs_uni_time = find_uni_num_msgs_uni_time_per(top_five_senders,
                                                                   recip_sender_time_sorted)
    graph_tsenders_uni_msgs(tsender_uni_num_msgs_uni_time)


if __name__ == "__main__":
    
    TOP_FIVE_SENDERS, SENDERS_NUM_MSGS_PER_TIME, SENDERS_TIME, PARSE_RECIP_DF = exec_q1()
    exec_q2(TOP_FIVE_SENDERS, SENDERS_NUM_MSGS_PER_TIME)
    exec_q3(TOP_FIVE_SENDERS, SENDERS_TIME, PARSE_RECIP_DF)
