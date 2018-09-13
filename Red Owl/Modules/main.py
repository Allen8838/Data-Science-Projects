import pandas as pd 
import csv 
import time 
#modules for question 1
from data_cleaning import place_names_in_original_and_cleaned_names_in_lists, replace_messy_names_w_cleaned_names, convert_names_to_lowercase, convert_unix_time_to_utc_time
from senders import create_dict_of_senders_n_count_num_msgs
from recipients import count_messages_received_by_each_recipient, parse_recipients, create_column_headers
from senders_recipients import find_num_msgs_sent_received_by_person
from collections import OrderedDict
import matplotlib.cbook

#module for question 2
from plot import graph_top_senders

#modules for question 3
from recipients import collect_columns_n_rows_top_senders_in_recipient
from senders_recipients import create_tuple_recipient_sender_time
from plot import create_list_by_person_unique_num_msgs_unique_time, graph_top_senders_uni_messages_over_time

def execute_procedure_for_question_1():
    #variables used
    list_of_dict_keys_for_num_msgs_sent_n_received_by_person = []
    number_of_senders_to_look = 5
    list_of_top_five_senders = []

    #read in the dataset given and add headers
    df = pd.read_csv(r'enron-event-history-all.csv', names=['time', 'message_id', 'sender', 'recipients', 'topic', 'mode'])
    #read in dataset created to clean the dataset given file
    df_dict_to_clean_names = pd.read_csv(r'Dictionary-to-clean-names.csv')

    #some sender names are not in a consistent format. placing sender names in original file as well as a cleanup name list into lists. lists are easier to work with then dataframes
    list_of_names_in_original_file, list_of_cleaned_names = place_names_in_original_and_cleaned_names_in_lists(df_dict_to_clean_names)
    df = replace_messy_names_w_cleaned_names(df, list_of_names_in_original_file, list_of_cleaned_names, 'sender')
    #converting all names to the same case so that same names of different cases will be grouped together
  
    df = convert_names_to_lowercase(df, 'sender')
    #converting the time from milliseconds to date (without time). having just the date will make it easier to graph the data for questions 2 and 3
    df = convert_unix_time_to_utc_time(df, 'time')

    #dict_senders_number_msgs_per_time will be used for question 2. 
    #list_of_senders_time will be used for question 3
    dict_for_number_msgs_sent_by_sender, dict_senders_number_msgs_per_time, list_of_senders_time = create_dict_of_senders_n_count_num_msgs(df)

    #sort by value in descending order. this will then be placed directly onto the CSV file. we will also know who are the top 5 senders from sorting this first
    dict_for_number_msgs_sent_by_sender_descending = OrderedDict(sorted(dict_for_number_msgs_sent_by_sender.items(), key=lambda t: t[1], reverse=True))

    dataframe_from_parse_recipients = helper_function_clean_dataframe_from_parse_recipients(df, list_of_names_in_original_file, list_of_cleaned_names)
    
    dict_for_msgs_received_by_recipient = count_messages_received_by_each_recipient(dataframe_from_parse_recipients) 

    #size of dict_for_msgs_received_by_recipient is larger than size of dict_for_number_msgs_sent_by_sender_descending. need to union the keys of these two dictionaries
    #the top 5 senders will be in this dictionary, so need to append the list with this dictionary first
    for key in dict_for_number_msgs_sent_by_sender_descending:
        #getting the union of the keys in both dictionaries, so will only append if the key DOESN'T already exist in the list
        if key not in list_of_dict_keys_for_num_msgs_sent_n_received_by_person:
            list_of_dict_keys_for_num_msgs_sent_n_received_by_person.append(key)

    for key in dict_for_msgs_received_by_recipient:
        if key not in list_of_dict_keys_for_num_msgs_sent_n_received_by_person:
            list_of_dict_keys_for_num_msgs_sent_n_received_by_person.append(key)

    #creates the file asked for Question 1 and returns a list of the top 5 senders for question 2
    list_of_top_five_senders = find_num_msgs_sent_received_by_person(dict_for_number_msgs_sent_by_sender_descending, dict_for_msgs_received_by_recipient, list_of_dict_keys_for_num_msgs_sent_n_received_by_person, number_of_senders_to_look)

    #return list_of_top_five_senders, dict_senders_number_msgs_per_time for question 2 
    #return list_of_senders_time and dataframe_from_parse_recipients for question 3
    return list_of_top_five_senders, dict_senders_number_msgs_per_time, list_of_senders_time, dataframe_from_parse_recipients

#perform datacleaning of dataframe_from_parse_recipients. should speed up execution of
#question 1 as the memory generated from this helper function will be released upon exit of the function
def helper_function_clean_dataframe_from_parse_recipients(dataframe, list_of_names_in_original_file, list_of_cleaned_names):
    dataframe_from_parse_recipients = parse_recipients(dataframe)

    #loop through each cell in the dataframe_from_parse_recipients and clean the names like what was done above
    #add headers to the dataframe first
    number_of_columns = len(dataframe_from_parse_recipients.columns)

    column_indexes = create_column_headers(number_of_columns)
    dataframe_from_parse_recipients.reset_index()
    dataframe_from_parse_recipients.columns = column_indexes

    dataframe_from_parse_recipients = replace_messy_names_w_cleaned_names(dataframe_from_parse_recipients, list_of_names_in_original_file, list_of_cleaned_names)

    return dataframe_from_parse_recipients

def execute_procedure_for_question_2(list_of_top_five_senders, dict_senders_number_msgs_per_time):
    graph_top_senders(list_of_top_five_senders, dict_senders_number_msgs_per_time)


def execute_procedure_for_question_3(list_of_top_five_senders, list_of_senders_time, dataframe_from_parse_recipients):
    modified_recipient_dataframe, list_of_column_n_rows_w_top_senders = collect_columns_n_rows_top_senders_in_recipient(dataframe_from_parse_recipients, list_of_top_five_senders)
    tuple_list_recipient_sender_time_sorted = create_tuple_recipient_sender_time(list_of_senders_time, modified_recipient_dataframe, list_of_column_n_rows_w_top_senders, list_of_top_five_senders)
    list_top_sender_uni_num_msgs_uni_time = create_list_by_person_unique_num_msgs_unique_time(list_of_top_five_senders, tuple_list_recipient_sender_time_sorted)
    graph_top_senders_uni_messages_over_time(list_top_sender_uni_num_msgs_uni_time)


if __name__ == "__main__":
    list_of_top_five_senders, dict_senders_number_msgs_per_time, list_of_senders_time, dataframe_from_parse_recipients = execute_procedure_for_question_1()
    execute_procedure_for_question_2(list_of_top_five_senders, dict_senders_number_msgs_per_time)
    execute_procedure_for_question_3(list_of_top_five_senders, list_of_senders_time, dataframe_from_parse_recipients)
    

