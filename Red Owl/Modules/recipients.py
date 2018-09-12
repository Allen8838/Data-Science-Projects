import pandas as pd 
from collections import defaultdict, Counter


dictionary_for_messages_received = defaultdict(int)

list_of_recipients = []

add_to_list = lambda x: list_of_recipients.append(str(x)) if str(x) != 'nan' else None

def count_messages_received_by_each_recipient(dataframe_from_parse_recipients):
    number_of_columns = len(dataframe_from_parse_recipients.columns)

    #storing recipients into a list so that we do not have to loop through a dataframe which is time costly
    for i in range(number_of_columns):
        dataframe_from_parse_recipients[i].map(add_to_list)

    #mirror image of the dictionary for number of messages the senders sent. e.g. if john adams was in 2 rows of the recipients column, john adams received a message twice
    for recipient in list_of_recipients:
        dictionary_for_messages_received[recipient] +=1

    return dictionary_for_messages_received, list_of_recipients




parse_recipient_by_pipe = lambda x: pd.Series([i for i in reversed(str(x).split('|'))])

def parse_recipients(dataframe):
    recipients_list = dataframe['recipients'].apply(parse_recipient_by_pipe)

    return recipients_list
