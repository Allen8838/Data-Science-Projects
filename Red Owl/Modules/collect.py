from dataframe_processing import convert_false_if_not_top_sender

def collect_columns_rows_with_top_senders(df_parse_recip, top_senders):
    """
    collects the cell value (column and row indexes), where the top five
    senders are located in the parsed recipient dataframe
    """

    column_rows_w_top_senders = []
    rows_of_interest = []

    # index to be used on dataframe so that we delete unnecessary columns based on name rather
    # than number, which may get shifted as we are deleting the column
    column_indexes = df_parse_recip.columns.values

    for column in df_parse_recip:
        df_parse_recip[column] = df_parse_recip[column].apply(convert_false_if_not_top_sender, args=(top_senders,))

    for index in column_indexes:
        if any(df_parse_recip[index]):
            # get the row index value where the top senders occur
            rows_of_interest = df_parse_recip.index[df_parse_recip[index] != False].tolist()
            column_rows_w_top_senders.append((index, rows_of_interest))

    return column_rows_w_top_senders
