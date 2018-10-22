"""
main module to run submodules
"""

import sys

from data_cleaning import get_preprocessed_input_data
from count_messages import count_msgs_received_by_each_recip,\
                           count_num_msgs_sent_by_sender,\
                           count_num_msgs_sent_at_one_time
from dataframe_processing import parse_recipients
from write import write_to_csv
from dictionary import create_union_of_keys
from find import find_top_senders_and_unique_recipients
from plot import graph_top_senders, graph_top_senders_with_unique_msgs

def create_csv_and_return_top_senders(df, dict_orig_cleaned_names, num_senders_to_look):
    """
    creates the csv file for question 1 and returns top senders, which we
    will need for questions 2 and 3
    """
    # find the number of messages sent by sender
    num_msgs_sent_by_sender = count_num_msgs_sent_by_sender(df)

    # find the number of messages received by recipient
    parse_recip_df = parse_recipients(df, dict_orig_cleaned_names)

    msgs_received_by_recip = count_msgs_received_by_each_recip(parse_recip_df)

    union_of_all_keys = create_union_of_keys(num_msgs_sent_by_sender, msgs_received_by_recip)

    # creates the file asked for Question 1
    write_to_csv(num_msgs_sent_by_sender,
                 msgs_received_by_recip,
                 union_of_all_keys)

    # return a list of the top senders for Question 2
    top_senders = union_of_all_keys[:num_senders_to_look]

    return top_senders

def visualize_number_msgs_sent_by_top_senders(senders_time, top_senders):
    """
    create graph over time of the top senders and number of messages sent
    """
    num_msgs_sent_at_one_time = count_num_msgs_sent_at_one_time(senders_time)

    graph_top_senders(top_senders, num_msgs_sent_at_one_time)


def visualize_unique_emails_received_by_top_senders(df, top_senders, dict_orig_cleaned_names):
    """
    create graph over time of the top senders and
    number of unique recipient emails received
    """
    top_sender_uni_num_msgs_uni_time = find_top_senders_and_unique_recipients(df,
                                                                              senders_time,
                                                                              TOP_SENDERS,
                                                                              dict_orig_cleaned_names)
    graph_top_senders_with_unique_msgs(top_sender_uni_num_msgs_uni_time)


if __name__ == "__main__":
    CSV_FILE = sys.argv[1]
    df, dict_orig_cleaned_names, senders_time = get_preprocessed_input_data(CSV_FILE)

    # Answer to Question 1
    TOP_SENDERS = create_csv_and_return_top_senders(df, dict_orig_cleaned_names, 5)

    # Answer to Question 2
    visualize_number_msgs_sent_by_top_senders(senders_time, TOP_SENDERS)

    # Answer to Question 3 - Need a helper function first before graphing
    visualize_unique_emails_received_by_top_senders(df, TOP_SENDERS, dict_orig_cleaned_names)
