from collections import defaultdict, Counter, OrderedDict

def count_num_msgs_sent_by_sender(df):
    senders = df['sender'].tolist()

    num_msgs_sent_by_sender = defaultdict(int)

    for sender in senders:
        num_msgs_sent_by_sender[sender] += 1

    # sort by value in descending order. this will then be placed directly onto the CSV file.
    # we will also know who are the top 5 senders from sorting this first
    num_msgs_sent_sender_desc = OrderedDict(sorted(num_msgs_sent_by_sender.items(),
                                                   key=lambda t: t[1],
                                                   reverse=True))

    return num_msgs_sent_sender_desc

def count_msgs_received_by_each_recip(df_parse_recip):
    """
    move the recipients from the dataframe into a list first
    to make it easier to work with. use dict to count msgs received
    by recipient
    """
    messages_received_by_recip = defaultdict(int)
    # storing recipients into a list so that we do not have to
    # loop through a dataframe which is time costly
    for column in df_parse_recip:
        tmp_list = df_parse_recip[column].tolist()
        tmp_list = [str(word) for word in tmp_list if str(word) != 'nan']
        # mirror image of the dictionary for number of messages the senders sent.
        # e.g. if john adams was in 2 rows of the recipients column,
        # john adams received a message twice
        for recipient in tmp_list:
            messages_received_by_recip[recipient] += 1

    return messages_received_by_recip


def count_num_msgs_sent_at_one_time(senders_time):
    count_senders_date_messages = Counter(tuple_pair for tuple_pair in senders_time)

    num_msgs_sent_at_one_time = defaultdict(list)

    for sender_time in count_senders_date_messages:
        num_msgs_sent_at_one_time[sender_time[0]].append((count_senders_date_messages[sender_time], sender_time[1]))

    return num_msgs_sent_at_one_time
