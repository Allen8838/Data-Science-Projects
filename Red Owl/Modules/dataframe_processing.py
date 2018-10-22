from data_cleaning import replace_messy_w_clean_names
from cell_processing import create_column_headers

def parse_recipients(dataframe, dict_orig_cleaned_names):
    """
    parse the recipient column of the dataframe by pipe
    """

    parse_recip_df = dataframe['recipients'].str.split('|', -1, expand=True)

    number_of_columns = len(parse_recip_df.columns)

    column_indexes = create_column_headers(number_of_columns)
    parse_recip_df.reset_index()
    parse_recip_df.columns = column_indexes

    parse_recip_df = replace_messy_w_clean_names(parse_recip_df, dict_orig_cleaned_names)

    return parse_recip_df


def del_columns_with_false_values(df_parse_recip, top_senders):
    """
    modifies the parsed recipient dataframe to delete columns where the rows equals False
    """
    
    # index to be used on dataframe so that we delete unnecessary columns based on name rather
    # than number, which may get shifted as we are deleting the column
    column_indexes = df_parse_recip.columns.values

    for column in df_parse_recip:
        df_parse_recip[column] = df_parse_recip[column].apply(convert_false_if_not_top_sender, args=(top_senders,))

    for index in column_indexes:
        if any(df_parse_recip[index]):
            continue
        else:
            # delete column that is full of False values
            del df_parse_recip[index]

    return df_parse_recip

def convert_false_if_not_top_sender(cell_value, top_senders):
    """
    helper function - make cell value false if it is not a top sender.
    allows later analysis to quickly skip any cell with false values.
    """

    if cell_value not in top_senders:
        return False
    return cell_value
