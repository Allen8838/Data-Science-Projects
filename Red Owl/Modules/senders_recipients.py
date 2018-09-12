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
