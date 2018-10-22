"""Function performing analyses on both senders and recipients"""

import csv

def find_top_senders(num_senders_desc, msgs_rcvd, both_dict_key, num_senders_look):
    """find the top senders (number specified by num_senders_look)"""

    file_for_question_1 = open('file_for_question_1.csv', 'w')
    top_five_senders = []

    with file_for_question_1:
        writer = csv.writer(file_for_question_1, lineterminator='\n')
        writer.writerow(['person', 'sent', 'received'])
        for i, key in enumerate(both_dict_key):
            if i < num_senders_look:
                #create a list of the top senders
                top_five_senders.append(key)
            try:
                #number of messages sent by a sender. need this try except clause
                #because not the list of keys may have people that have received
                #an email and did not send an email and vice versa
                messages_sent = num_senders_desc[key]
            except:
                messages_sent = 0
            try:
                #number of messages received by a sender
                messages_received = msgs_rcvd[key]
            except:
                messages_received = 0

            writer.writerow((key, messages_sent, messages_received))

    return top_five_senders

def create_recip_sender_time(senders_time, recip_df, col_rows_w_tsenders, top_five_senders):
    """create pairings of who sent an email to a recipient and at what time"""
    #tsenders stands for top senders
    recip_sender_time = []

    for column_row in col_rows_w_tsenders:
        column = column_row[0]
        row_list = column_row[1]
        for row in row_list:
            sender_time = senders_time[row]
            recipient = recip_df.at[row, column]
            recipient_sender_time = (recipient,)+sender_time
            recip_sender_time.append(recipient_sender_time)

    #create a set for recip_sender_time in case a recipient received multiple
    #emails from the same sender at the same time
    recip_sender_time = list(set(recip_sender_time))

    recip_sender_time_sorted = sort_recip_time_by_person(top_five_senders, recip_sender_time)

    return recip_sender_time_sorted


def sort_recip_time_by_person(top_five_senders, recip_sender_time):
    """will re-sort the list from above and group them first by how they appear in the
    list of top senders. this will make it easier when we loop through and plot later"""

    recip_sender_time_sorted = []

    for sender in top_five_senders:
        #appending this so that each top sender is in their own list, making
        #it easier to iterate and create unique lists based on top senders in the
        #procedure create_list_by_person_unique_num_msgs_unique_time. since
        #we are counting the top five senders, this will create 5 lists.
        #although there is an if startswith statement below, it doesn't really
        #reduce our recip_sender_time as we had initially filtered
        #top senders in the recipients dataframe
        recip_sender_time_sorted.append([tuple_three for tuple_three in recip_sender_time if tuple_three[0].startswith(sender)])

    return recip_sender_time_sorted
