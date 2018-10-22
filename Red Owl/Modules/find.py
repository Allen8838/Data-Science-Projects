from create_tuples import create_tuple_senders_time, create_recip_sender_time
from dataframe_processing import parse_recipients
from collect import collect_columns_rows_with_top_senders
from dataframe_processing import del_columns_with_false_values
from collections import defaultdict

def find_top_senders_and_unique_recipients(df, senders_time, top_senders, dict_orig_cleaned_names):
    """
    perform additional processing of the expanded recipient dataframe columns.
    find top senders and the unique number of recipients they sent messages to and
    graph results
    """
    # separate each recipient by pipe
    parse_recip_df = parse_recipients(df, dict_orig_cleaned_names)

    # find where the top senders are located in expanded df and 
    # mark all other cells as false
    column_rows_w_top_senders = collect_columns_rows_with_top_senders(parse_recip_df,
                                                                      top_senders)
    # truncate df to only the ones we need
    mod_recip_df = del_columns_with_false_values(parse_recip_df, top_senders)

    # create pairings of who sent an email to a recipient and at what time
    recip_sender_time_sorted = create_recip_sender_time(senders_time,
                                                        mod_recip_df,
                                                        column_rows_w_top_senders,
                                                        top_senders)

    top_sender_unique_num_msgs_unique_time = find_unique_num_msgs_unique_time_per_person(top_senders,
                                                                   recip_sender_time_sorted)
    
    return top_sender_unique_num_msgs_unique_time

def find_unique_num_msgs_unique_time_per_person(top_senders, tuple_recip_sender_time_sorted):
    """
    helper function - create a list keeping track of the unique people that
    have sent a message to the top sender and at what date
    """

    # this will at the end look like, for example
    # [(jeff dasovich, [3,4,2], [7/11/2001, 7/12/2001, 7/20/2001])]
    top_senders_unique_num_msgs_unique_time = []

    for i, _ in enumerate(top_senders):
        name_of_sender = top_senders[i]
        # keep track of who has sent an email to the top sender. if already seen, do not append
        tmp_list_of_seen_senders = []
        # will use a dictionary where the key will be the date and
        # the value will be the count of unique emails on that date
        tmp_dict = defaultdict(int)
        list_for_a_top_sender = tuple_recip_sender_time_sorted[i]
        for tuple_three in list_for_a_top_sender:
            if tuple_three[1] not in tmp_list_of_seen_senders:
                tmp_list_of_seen_senders.append(tuple_three[1])
                tmp_dict[tuple_three[2]] += 1

        top_senders_unique_num_msgs_unique_time.append((name_of_sender, 
                                                        list(tmp_dict.values()), 
                                                        list(tmp_dict.keys())))


    return top_senders_unique_num_msgs_unique_time
