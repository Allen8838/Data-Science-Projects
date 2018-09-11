import pandas as pd 
import csv 
#modules for question 1
from data_cleaning import place_names_in_original_and_cleaned_names_in_lists, replace_messy_names_w_cleaned_names, convert_names_to_lowercase, convert_unix_time_to_utc_time
from senders import create_dictionary_of_senders_n_count_num_msgs
from recipients import count_messages_received_by_each_recipient, parse_recipients
from senders_recipients import find_number_msgs_sent_received_by_person
from collections import OrderedDict
import gc
import objgraph 

import warnings
import matplotlib.cbook

from memory_profiler import profile
import os
import psutil
process = psutil.Process(os.getpid())

#module for question 2
from plot import graph_top_senders

#modules for question 3


#data preprocessing. add a header row for the column names. 


@profile 
def execute_procedure_for_question_1():
    #variables used
    list_of_dict_keys_for_num_msgs_sent_n_received_by_person = []

    number_of_senders_to_look = 5

    list_of_top_five_senders = []

    #read in the dataset given and the dataset created for cleaning
    df = pd.read_csv(r'enron-event-history-all.csv', names=['time', 'message_id', 'sender', 'recipients', 'topic', 'mode'])

    df_dictionary_to_clean_names = pd.read_csv(r'Dictionary to clean names.csv')

    #some sender names are not in a consistent format. placing sender names in original file as well as a cleanup name list into lists. lists are easier to work with then dataframes
    list_of_names_in_original_file, list_of_cleaned_names = place_names_in_original_and_cleaned_names_in_lists(df_dictionary_to_clean_names)

    df = replace_messy_names_w_cleaned_names(df, list_of_names_in_original_file, list_of_cleaned_names)

    #converting all names to the same case so that same names of different cases will be grouped together
    df = convert_names_to_lowercase(df, 'sender')

    #converting the time from milliseconds to date (without time). having just the date will make it easier to graph the data for questions 2 and 3
    df = convert_unix_time_to_utc_time(df, 'time')

    #dictionary_senders_number_msgs_per_time will be used for question 2. 
    dictionary_for_number_msgs_sent_by_sender, dictionary_senders_number_msgs_per_time, _ = create_dictionary_of_senders_n_count_num_msgs(df)

    #sort by value in descending order. this will then be placed directly onto the CSV file
    dictionary_for_number_msgs_sent_by_sender_descending = OrderedDict(sorted(dictionary_for_number_msgs_sent_by_sender.items(), key=lambda t: t[1], reverse=True))

    dataframe_from_parse_recipients = parse_recipients(df)
    dictionary_for_messages_received, _ = count_messages_received_by_each_recipient(dataframe_from_parse_recipients) 

    #size of dictionary_for_messages_received is larger than size of dictionary_for_number_msgs_sent_by_sender_descending. need to union the keys of these two dictionaries
    for key in dictionary_for_number_msgs_sent_by_sender_descending:
        #getting the union of the keys in both dictionaries, so will only append if the key DOESN'T already exist in the list
        if key not in list_of_dict_keys_for_num_msgs_sent_n_received_by_person:
            list_of_dict_keys_for_num_msgs_sent_n_received_by_person.append(key)

    for key in dictionary_for_messages_received:
        if key not in list_of_dict_keys_for_num_msgs_sent_n_received_by_person:
            list_of_dict_keys_for_num_msgs_sent_n_received_by_person.append(key)

    #creates the file asked for Question 1 and returns a list of the top 5 senders for question 2
    list_of_top_five_senders = find_number_msgs_sent_received_by_person(dictionary_for_number_msgs_sent_by_sender_descending, dictionary_for_messages_received, list_of_dict_keys_for_num_msgs_sent_n_received_by_person, number_of_senders_to_look)

    
    return list_of_top_five_senders, dictionary_senders_number_msgs_per_time

@profile 
def execute_procedure_for_question_2(list_of_top_five_senders, dictionary_senders_number_msgs_per_time):
    #print(objgraph.show_most_common_types())
    #warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
    #procedures for Question 2
    graph_top_senders(list_of_top_five_senders, dictionary_senders_number_msgs_per_time)


if __name__ == "__main__":
    list_of_top_five_senders, dictionary_senders_number_msgs_per_time = execute_procedure_for_question_1()
    execute_procedure_for_question_2(list_of_top_five_senders, dictionary_senders_number_msgs_per_time)

