import pandas as pd 
from collections import defaultdict, Counter


dictionary_for_msgs_received_by_recipient = defaultdict(int)

list_of_recipients = []

add_to_list = lambda x: list_of_recipients.append(str(x)) if str(x) != 'nan' else None

def count_messages_received_by_each_recipient(dataframe_from_parse_recipients):
    number_of_columns = len(dataframe_from_parse_recipients.columns)

    #storing recipients into a list so that we do not have to loop through a dataframe which is time costly
    for i in range(number_of_columns):
        dataframe_from_parse_recipients[i].map(add_to_list)

    #mirror image of the dictionary for number of messages the senders sent. e.g. if john adams was in 2 rows of the recipients column, john adams received a message twice
    # for recipient in list_of_recipients:
    #     dictionary_for_msgs_received_by_recipient[recipient] +=1

    return dictionary_for_msgs_received_by_recipient




parse_recipient_by_pipe = lambda x: pd.Series([i for i in reversed(str(x).split('|'))])

def parse_recipients(dataframe):
    recipients_list = dataframe['recipients'].apply(parse_recipient_by_pipe)

    return recipients_list




list_of_columns_to_delete = []
list_of_column_n_rows_w_top_senders = []
rows_of_interest = []
    

def convert_cell_value_false_if_not_top_sender(cell_value, list_of_top_five_senders):
    if cell_value not in list_of_top_five_senders:
        return False
    else:
        return cell_value

def collect_columns_n_rows_top_senders_in_recipient(dataframe, list_of_top_five_senders):
    recipient_dataframe = dataframe['recipients'].apply(parse_recipient_by_pipe)

    number_of_columns = len(recipient_dataframe.columns)
    column_indexes = []
    #index to be used on dataframe so that we delete unnecessary columns based on name rather than number, which may get shifted as we are deleting the column
    for i in range(number_of_columns):
        recipient_dataframe[i] = recipient_dataframe[i].apply(convert_cell_value_false_if_not_top_sender, args=(list_of_top_five_senders,))
        column_indexes.append('a'+str(i))
    
    recipient_dataframe.reset_index()
    recipient_dataframe.columns = column_indexes

    for index in column_indexes:
        if any(recipient_dataframe[index]):
            #get the row index value where the top senders occur
            rows_of_interest = recipient_dataframe.index[recipient_dataframe[index] != False].tolist()
            list_of_column_n_rows_w_top_senders.append((index, rows_of_interest))
        else:
            del recipient_dataframe[index]
    
    return recipient_dataframe, list_of_column_n_rows_w_top_senders
    #return recipient_dataframe.loc[recipient_dataframe[0] != False]
