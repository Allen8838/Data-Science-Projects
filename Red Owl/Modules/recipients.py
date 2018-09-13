import pandas as pd 
from collections import defaultdict, Counter


dictionary_for_msgs_received_by_recipient = defaultdict(int)

list_of_recipients = []
add_to_list = lambda x: list_of_recipients.append(str(x)) if str(x) != 'nan' else None

def count_messages_received_by_each_recipient(dataframe_from_parse_recipients):
    #storing recipients into a list so that we do not have to loop through a dataframe which is time costly
    for column in dataframe_from_parse_recipients:
        dataframe_from_parse_recipients[column].map(add_to_list)

    #mirror image of the dictionary for number of messages the senders sent. e.g. if john adams was in 2 rows of the recipients column, john adams received a message twice
    for recipient in list_of_recipients:
        dictionary_for_msgs_received_by_recipient[recipient] +=1

    return dictionary_for_msgs_received_by_recipient


parse_recipient_by_pipe = lambda x: pd.Series([i for i in reversed(str(x).split('|'))])

def parse_recipients(dataframe):
    parsed_recipients_dataframe = dataframe['recipients'].apply(parse_recipient_by_pipe)

    return parsed_recipients_dataframe
  

def convert_cell_value_false_if_not_top_sender(cell_value, list_of_top_five_senders):
    if cell_value not in list_of_top_five_senders:
        return False
    else:
        return cell_value

def create_column_headers(number_of_columns):
    column_indexes = []
    #index to be used on dataframe so that we delete unnecessary columns based on name rather than number, which may get shifted as we are deleting the column
    for i in range(number_of_columns):
        column_indexes.append('a'+str(i))
    
    return column_indexes

def collect_columns_n_rows_top_senders_in_recipient(dataframe_from_parse_recipients, list_of_top_five_senders):
    list_of_column_n_rows_w_top_senders = []
    rows_of_interest = []

    number_of_columns = len(dataframe_from_parse_recipients.columns)
    
    #index to be used on dataframe so that we delete unnecessary columns based on name rather than number, which may get shifted as we are deleting the column
    column_indexes = dataframe_from_parse_recipients.columns.values

    for column in dataframe_from_parse_recipients:
        dataframe_from_parse_recipients[column] = dataframe_from_parse_recipients[column].apply(convert_cell_value_false_if_not_top_sender, args=(list_of_top_five_senders,))
        
    for index in column_indexes:
        if any(dataframe_from_parse_recipients[index]):
            #get the row index value where the top senders occur
            rows_of_interest = dataframe_from_parse_recipients.index[dataframe_from_parse_recipients[index] != False].tolist()
            list_of_column_n_rows_w_top_senders.append((index, rows_of_interest))
        else:
            #delete column that is full of False values
            del dataframe_from_parse_recipients[index]
    
    return dataframe_from_parse_recipients, list_of_column_n_rows_w_top_senders
   
