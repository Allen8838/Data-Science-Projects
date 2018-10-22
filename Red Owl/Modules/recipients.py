"""Function performing analyses of recipients"""

from collections import defaultdict
from data_cleaning import replace_messy_w_clean_names

def cnt_msgs_recvd_by_each_recip(df_parse_recip):
    """move the recipients from the dataframe into a list first
    to make it easier to work with. use dict to count msgs received
    by recipient"""
    msgs_rcvd_by_recipient = defaultdict(int)
    #storing recipients into a list so that we do not have to
    #loop through a dataframe which is time costly
    for column in df_parse_recip:
        tmp_list = df_parse_recip[column].tolist()
        tmp_list = [str(word) for word in tmp_list if str(word) != 'nan']
        #mirror image of the dictionary for number of messages the senders sent.
        #e.g. if john adams was in 2 rows of the recipients column,
        #john adams received a message twice
        for recipient in tmp_list:
            msgs_rcvd_by_recipient[recipient] += 1

    return msgs_rcvd_by_recipient

def convert_false_if_not_tsender(cell_value, top_five_senders):
    """make cell value false if it is not a top sender.
    allows later analysis to quickly skip any cell with false values.
    this function placed in this module because want to know how
    many msgs top senders received"""

    if cell_value not in top_five_senders:
        return False
    return cell_value

def create_column_headers(number_of_columns):
    """creates a column header when recipient column is parsed.
    makes the column header based on string value rather than number,
    which may get shifted as we are deleting the column in a later analysis"""

    column_indexes = []
    for i in range(number_of_columns):
        column_indexes.append('a'+str(i))
    return column_indexes

def coll_cols_rows_tsenders_recip(df_parse_recip, top_five_senders):
    """collects the cell value (column and row indexes), where the top five
    senders are located in the parsed recipient dataframe. modifies the parsed recipient
    dataframe to delete columns where the rows equals False"""

    col_n_rows_w_top_senders = []
    rows_of_interest = []

    #index to be used on dataframe so that we delete unnecessary columns based on name rather
    #than number, which may get shifted as we are deleting the column
    column_indexes = df_parse_recip.columns.values

    for column in df_parse_recip:
        df_parse_recip[column] = df_parse_recip[column].apply(convert_false_if_not_tsender, args=(top_five_senders,))

    for index in column_indexes:
        if any(df_parse_recip[index]):
            #get the row index value where the top senders occur
            rows_of_interest = df_parse_recip.index[df_parse_recip[index] != False].tolist()
            col_n_rows_w_top_senders.append((index, rows_of_interest))
        else:
            #delete column that is full of False values
            del df_parse_recip[index]

    return df_parse_recip, col_n_rows_w_top_senders


def parse_recipients(dataframe, dict_orig_cleaned_names):
    """parse the recipient column of the dataframe by pipe."""

    parse_recip_df = dataframe['recipients'].str.split('|', -1, expand=True)

    number_of_columns = len(parse_recip_df.columns)

    column_indexes = create_column_headers(number_of_columns)
    parse_recip_df.reset_index()
    parse_recip_df.columns = column_indexes

    parse_recip_df = replace_messy_w_clean_names(parse_recip_df, dict_orig_cleaned_names)

    return parse_recip_df
