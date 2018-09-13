import pandas as pd 
from collections import defaultdict, Counter


dictionary_for_number_msgs_sent_by_sender = defaultdict(int)
dictionary_senders_list_of_time = defaultdict(list)
dictionary_senders_number_msgs_per_time = defaultdict(list)

list_of_senders = []
list_of_time = []
list_of_senders_time = []

add_to_list_of_senders = lambda x: list_of_senders.append(str(x)) if x != '' and x != 'sender' else None
add_to_list_of_time = lambda x: list_of_time.append(x) if x != '' and x != 'time' else None

def create_dict_of_senders_n_count_num_msgs(dataframe):
    #putting the dataframe column into a list through map as looping through each row in the dataframe is time consuming
    dataframe['sender'].map(add_to_list_of_senders)
    dataframe['time'].map(add_to_list_of_time)

    list_of_senders_time = zip(list_of_senders, list_of_time)

    unzipped_list_of_senders_time = list(list_of_senders_time)

    #create a "dictionary" of counts instances of the same sender time pair
    #resulting dictionary will look something like e.g. ('trnews tr', '2002-11-30'): 1, ('hussey eudoramail', '2002-12-21'): 1}
    count_senders_date_messages = Counter(tuple_pair for tuple_pair in unzipped_list_of_senders_time)

    #count the number of messages per sender
    for sender in list_of_senders:
        dictionary_for_number_msgs_sent_by_sender[sender] += 1

    #Need this dictionary for question 2
    for sender_time in count_senders_date_messages:
        dictionary_senders_number_msgs_per_time[sender_time[0]].append((count_senders_date_messages[sender_time], sender_time[1]))

    
    return dictionary_for_number_msgs_sent_by_sender, dictionary_senders_number_msgs_per_time, unzipped_list_of_senders_time