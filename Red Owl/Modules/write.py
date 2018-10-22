import csv

def write_to_csv(num_senders_desc, msgs_rcvd, both_dict_key):
    """
    create a csv of person, sent and received columns
    """

    file_for_question_1 = open('file_for_question_1.csv', 'w')

    with file_for_question_1:
        writer = csv.writer(file_for_question_1, lineterminator='\n')
        writer.writerow(['person', 'sent', 'received'])
        for key in both_dict_key:
            try:
                # number of messages sent by a sender. need this try except clause
                # because the list of keys may have people that have received
                # an email and did not send an email and vice versa
                messages_sent = num_senders_desc[key]
            except:
                messages_sent = 0
            try:
                # number of messages received by a sender
                messages_received = msgs_rcvd[key]
            except:
                messages_received = 0

            writer.writerow((key, messages_sent, messages_received))

    return None
