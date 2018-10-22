
def convert_false_if_not_top_sender(cell_value, top_five_senders):
    """
    make cell value false if it is not a top sender.
    allows later analysis to quickly skip any cell with false values.
    """

    if cell_value not in top_five_senders:
        return False
    return cell_value

def create_column_headers(number_of_columns):
    """
    creates a column header when recipient column is parsed.
    makes the column header based on string value rather than number,
    which may get shifted as we are deleting the column in a later analysis
    """

    column_indexes = []
    for i in range(number_of_columns):
        column_indexes.append('a'+str(i))
    return column_indexes
