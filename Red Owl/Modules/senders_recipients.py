import csv 

def find_num_msgs_sent_received_by_person(dictionary_for_number_of_senders_descending, dictionary_for_messages_received, list_of_both_dictionary_keys, number_of_senders_to_look):
    file_for_question_1 = open('file_for_question_1.csv', 'w')
    list_of_top_five_senders = []
    
    with file_for_question_1:
        writer = csv.writer(file_for_question_1, lineterminator = '\n')
        writer.writerow(['person', 'sent', 'received'])
        for i, key in enumerate(list_of_both_dictionary_keys):
            if i<number_of_senders_to_look:
                #create a list of the top senders
                list_of_top_five_senders.append(key)
            try: 
                #number of messages sent by a sender. need this try except clause because not the list of keys may have people
                #that have received an email and did not send an email and vice versa
                messages_sent = dictionary_for_number_of_senders_descending[key]
            except:
                messages_sent = 0
            try:
                #number of messages received by a sender
                messages_received = dictionary_for_messages_received[key]
            except:
                messages_received = 0 

            writer.writerow((key, messages_sent, messages_received))


    return list_of_top_five_senders

def create_tuple_recipient_sender_time(list_of_senders_time, recipient_dataframe_list, list_of_column_n_rows_w_top_senders, list_of_top_five_senders):
    list_recipient_sender_time = []

    for column_row in list_of_column_n_rows_w_top_senders:
        column = column_row[0]
        row_list = column_row[1]
        for row in row_list:
            sender_time = list_of_senders_time[row]
            recipient = recipient_dataframe_list.at[row, column]
            recipient_sender_time = (recipient,)+sender_time
            list_recipient_sender_time.append(recipient_sender_time)

    #create a set for list_recipient_sender_time in case a recipient received multiple emails from the same sender at the same time
    list_recipient_sender_time = list(set(list_recipient_sender_time))

    tuple_list_recipient_sender_time_sorted = create_new_list_of_tuple_recipient_time_by_person(list_of_top_five_senders, list_recipient_sender_time)
  
    return tuple_list_recipient_sender_time_sorted


#will resort the list from above and group them first by how they appear in the list of top senders. this will make it easier when we loop through and plot later
def create_new_list_of_tuple_recipient_time_by_person(list_of_top_five_senders, list_recipient_sender_time):
    tuple_list_recipient_sender_time_sorted = []

    for sender in list_of_top_five_senders:
        tuple_list_recipient_sender_time_sorted.append([tuple_three for tuple_three in list_recipient_sender_time if tuple_three[0].startswith(sender)])    #appending this so that each top sender is in their own list, making it easier to iterate and create unique lists based on top senders in the procedure create_list_by_person_unique_num_msgs_unique_time

    return tuple_list_recipient_sender_time_sorted
