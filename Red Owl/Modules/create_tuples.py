

def create_tuple_senders_time(df):
    """
    Create a dictionary of the number of messages a sender sent,
    and a dictionary that counts the number of messages a sender sent
    on a certain date
    """
    # putting the df column into a list through map as
    # looping through each row in the df is time consuming
    senders = df['sender'].tolist()

    time_ls = df['time'].tolist()

    # discard empty values and header names
    senders = [str(name) for name in senders if str(name) != '' and str(name) != 'sender']
    time_ls = [timestamp for timestamp in time_ls if timestamp != '' and timestamp != 'time']

    senders_time = zip(senders, time_ls)

    unzipped_senders_time = list(senders_time)

    return unzipped_senders_time


def create_recip_sender_time(senders_time, recip_df, col_rows_w_top_senders, top_five_senders):
    """
    create pairings of who sent an email to a recipient and at what time
    """

    recip_sender_time = []

    for column_row in col_rows_w_top_senders:
        column = column_row[0]
        row_list = column_row[1]
        for row in row_list:
            sender_time = senders_time[row]
            recipient = recip_df.at[row, column]
            recipient_sender_time = (recipient,)+sender_time
            recip_sender_time.append(recipient_sender_time)

    # create a set for recip_sender_time in case a recipient received multiple
    # emails from the same sender at the same time
    recip_sender_time = list(set(recip_sender_time))

    recip_sender_time_sorted = sort_recip_time_by_person(top_five_senders, recip_sender_time)

    return recip_sender_time_sorted


def sort_recip_time_by_person(top_five_senders, recip_sender_time):
    """
    helper function - will re-sort the list from above and group them first by how they appear in the
    list of top senders. this will make it easier when we loop through and plot later
    """

    recip_sender_time_sorted = []

    for sender in top_five_senders:
        # appending this so that each top sender is in their own list, making
        # it easier to iterate and create unique lists based on top senders in the
        # procedure create_list_by_person_unique_num_msgs_unique_time. since
        # we are counting the top five senders, this will create 5 lists.
        # although there is an if startswith statement below, it doesn't really
        # reduce our recip_sender_time as we had initially filtered
        # top senders in the recipients dataframe
        recip_sender_time_sorted.append([tuple_three for tuple_three in recip_sender_time if tuple_three[0].startswith(sender)])

    return recip_sender_time_sorted
