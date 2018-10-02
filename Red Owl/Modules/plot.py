"""Module to plot graphs for questions 2 and 3"""

from collections import defaultdict
import datetime
from matplotlib.dates import date2num
import matplotlib.pyplot as plt


def graph_top_senders(top_five_senders, dict_senders_num_msgs_per_time):
    """function for question 2. shows number of emails sent
    over time by the top senders"""
    number_of_subplots = len(top_five_senders)
    figure = plt.figure(figsize=(800/96, 800/96), dpi=96)

    for i, _ in enumerate(top_five_senders):
        name_of_sender = top_five_senders[i]
        list_of_time = dict_senders_num_msgs_per_time[name_of_sender]
        date_array = []
        number_messages_array = []
        for num_messages_n_date in list_of_time:
            date = num_messages_n_date[1].split(' ')[0]
            year = date.split('-')[0]
            month = date.split('-')[1]
            date_array.append(datetime.datetime(int(year), int(month), 1, 0))
            number_messages_array.append(num_messages_n_date[0])
            axes = plt.subplot(number_of_subplots, 1, i+1)
            axes.set_title('Emails sent over time by {}'.format(name_of_sender))
            axes.set_xlabel("Time in Month and Year")
            axes.set_ylabel("Number of Emails")
            axes.bar(date_array, number_messages_array, width=100)
            axes.xaxis_date()
            axes.set_anchor('W')

    figure.tight_layout()
    figure.savefig('Number of emails sent over time by the top {} senders.png'.format(number_of_subplots), dpi=figure.dpi)

def graph_tsenders_uni_msgs(tsender_uni_num_msgs_uni_time):
    """create graph for question 3 """
    name_senders = []
    uni_msgs = []
    uni_time = []

    figure = plt.figure()

    for sender_uni_msgs_uni_time in tsender_uni_num_msgs_uni_time:
        name_senders.append(sender_uni_msgs_uni_time[0])
        uni_msgs.append(sender_uni_msgs_uni_time[1])
        one_list_of_date = sender_uni_msgs_uni_time[2]

        date_array = []
        #need to make sure time is datetime object in order to plot graph
        for date in one_list_of_date:
            year = date.split('-')[0]
            month = date.split('-')[1]
            date_array.append(date2num(datetime.datetime(int(year), int(month), 1, 0)))

        uni_time.append(date_array)


    ax = plt.subplot(111)
    ax.bar(uni_time[0], uni_msgs[0], width=8, color='b', align='center', label=name_senders[0])
    ax.bar(uni_time[1], uni_msgs[1], width=8, color='g', align='center', label=name_senders[1])
    ax.bar(uni_time[2], uni_msgs[2], width=8, color='r', align='center', label=name_senders[2])
    ax.bar(uni_time[3], uni_msgs[3], width=8, color='c', align='center', label=name_senders[3])
    ax.bar(uni_time[4], uni_msgs[4], width=8, color='m', align='center', label=name_senders[4])
    ax.set_xlabel("Time in Month and Year")
    ax.set_ylabel("# of Unique emails received")
    ax.legend()
    ax.xaxis_date()
    figure.savefig('Unique emails received by top sender over time', dpi=figure.dpi)
    

def find_uni_num_msgs_uni_time_per(top_five_senders, tuple_recip_sender_time_sorted):
    """helper function to graph question 3"""

    #this will at the end look like, for example
    #[(jeff dasovich, [3,4,2], [7/11/2001, 7/12/2001, 7/20/2001])]
    tsender_uni_num_msgs_uni_time = []

    for i, _ in enumerate(top_five_senders):
        name_of_sender = top_five_senders[i]
        #keep track of who has sent an email to the top sender. if already seen, do not append
        tmp_list_of_seen_senders = []
        #will use a dictionary where the key will be the date and
        #the value will be the count of unique emails on that date
        tmp_dict = defaultdict(int)
        list_for_a_top_sender = tuple_recip_sender_time_sorted[i]
        for tuple_three in list_for_a_top_sender:
            #this should be the case but double checking that it is true
            if tuple_three[0] == name_of_sender:
                if tuple_three[1] not in tmp_list_of_seen_senders:
                    tmp_list_of_seen_senders.append(tuple_three[1])
                    tmp_dict[tuple_three[2]] += 1

        tsender_uni_num_msgs_uni_time.append((name_of_sender, list(tmp_dict.values()), list(tmp_dict.keys())))


    return tsender_uni_num_msgs_uni_time
