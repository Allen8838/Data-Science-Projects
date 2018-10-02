"""Functions on performing analyses on senders"""

from collections import defaultdict, Counter


def create_dict_senders_num_msgs(df):
    """Create a dictionary of the number of messages a sender sent,
    and a dictionary that counts the number of messages a sender sent
    on a certain date"""
    senders = []
    add_senders_ls = lambda x: senders.append(str(x)) if x != '' and x != 'sender' else None
    time_ls = []
    add_time_ls = lambda x: time_ls.append(x) if x != '' and x != 'time' else None

    #putting the df column into a list through map as
    # looping through each row in the df is time consuming
    df['sender'].map(add_senders_ls)
    df['time'].map(add_time_ls)

    senders_time = zip(senders, time_ls)

    unzipped_senders_time = list(senders_time)

    #create a "dictionary" of counts instances of the same sender time pair
    #resulting dictionary will look something like e.g. ('trnews tr', '2002-11-30'):
    # 1, ('hussey eudoramail', '2002-12-21'): 1}
    count_senders_date_messages = Counter(tuple_pair for tuple_pair in unzipped_senders_time)

    dict_num_msgs_sender_sent = defaultdict(int)
    #count the number of messages per sender
    for sender in senders:
        dict_num_msgs_sender_sent[sender] += 1

    dict_senders_num_msgs_per_time = defaultdict(list)
    #Need this dictionary for question 2
    for sender_time in count_senders_date_messages:
        dict_senders_num_msgs_per_time[sender_time[0]].append((count_senders_date_messages[sender_time], sender_time[1]))

    
    return dict_num_msgs_sender_sent, dict_senders_num_msgs_per_time, unzipped_senders_time
