from collections import defaultdict
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
import time
import numpy as np
import datetime
import pylab 

def graph_top_senders(list_of_top_five_senders, dict_senders_number_msgs_per_time):
    number_of_subplots = len(list_of_top_five_senders)
    figure = plt.figure(figsize=(800/96, 800/96), dpi=96)
    
    for i in range(len(list_of_top_five_senders)):
        name_of_sender = list_of_top_five_senders[i] 
        list_of_time = dict_senders_number_msgs_per_time[name_of_sender]
        date_array = []
        number_messages_array = []
        for num_messages_n_date in list_of_time:
            #date = datetime.datetime.strptime(num_messages_n_date[1], '%Y-%m-%d')  #converting the date from a regular string to datetime object so that graph will plot
            date = num_messages_n_date[1].split(' ')[0]
            year = date.split('-')[0]
            month = date.split('-')[1]
            date_array.append(datetime.datetime(int(year), int(month), 1, 0))
            number_messages_array.append(num_messages_n_date[0])
           # print(number_messages_array)
            axes = plt.subplot(number_of_subplots,1,i+1)
            axes.set_title('Emails sent over time by {}'.format(name_of_sender))
            axes.set_xlabel("Time in Month and Year")
            axes.set_ylabel("Number of Emails")
            axes.bar(date_array, number_messages_array, width=100)
            axes.xaxis_date()
            axes.set_anchor('W')
    
    figure.tight_layout()
    figure.savefig('Number of emails sent over time by the top {} senders.png'.format(number_of_subplots), dpi=figure.dpi)
    

def create_list_by_person_unique_num_msgs_unique_time(list_of_top_five_senders, tuple_list_recipient_sender_time_sorted):
    number_of_subplots = len(list_of_top_five_senders)

    list_top_sender_uni_num_msgs_uni_time = []  #this will at the end look for example [(jeff dasovich, [3,4,2], [7/11/2001, 7/12/2001, 7/20/2001])]

    for i in range(len(list_of_top_five_senders)):
        name_of_sender = list_of_top_five_senders[i]
        #keep track of who has sent an email to the top sender. if already seen, do not append
        tmp_list_of_seen_senders = []
        #will use a dictionary where the key will be the date and the value will be the count of unique emails on that date
        tmp_dict = defaultdict(int)
        list_for_a_top_sender = tuple_list_recipient_sender_time_sorted[i]
        for tuple_three in list_for_a_top_sender:
            #this should be the case but double checking that it is true
            if tuple_three[0] == name_of_sender:
                if tuple_three[1] not in tmp_list_of_seen_senders:
                    tmp_list_of_seen_senders.append(tuple_three[1])
                    tmp_dict[tuple_three[2]] += 1
 
        list_top_sender_uni_num_msgs_uni_time.append((name_of_sender, list(tmp_dict.values()), list(tmp_dict.keys())))
 
        

    return list_top_sender_uni_num_msgs_uni_time


def graph_top_senders_uni_messages_over_time(list_top_sender_uni_num_msgs_uni_time):
    number_of_subplots = len(list_top_sender_uni_num_msgs_uni_time)

    list_name_of_senders = []
    list_of_uni_msgs = []
    list_of_uni_time = []

    figure = plt.figure()

    for i, sender_uni_msgs_uni_time in enumerate(list_top_sender_uni_num_msgs_uni_time):
        list_name_of_senders.append(sender_uni_msgs_uni_time[0])
        list_of_uni_msgs.append(sender_uni_msgs_uni_time[1])
        one_list_of_date = sender_uni_msgs_uni_time[2]

        date_array = []
        #need to make sure time is datetime object in order to plot graph
        for date in one_list_of_date:
            year = date.split('-')[0]
            month = date.split('-')[1]
            date_array.append(date2num(datetime.datetime(int(year), int(month), 1, 0)))

        list_of_uni_time.append(date_array)


    ax = plt.subplot(111)
    ax.bar(list_of_uni_time[0], list_of_uni_msgs[0], width=8, color='b', align='center', label=list_name_of_senders[0])
    ax.bar(list_of_uni_time[1], list_of_uni_msgs[1], width=8, color='g', align='center', label=list_name_of_senders[1])
    ax.bar(list_of_uni_time[2], list_of_uni_msgs[2], width=8, color='r', align='center', label=list_name_of_senders[2])
    ax.bar(list_of_uni_time[3], list_of_uni_msgs[3], width=8, color='c', align='center', label=list_name_of_senders[3])
    ax.bar(list_of_uni_time[4], list_of_uni_msgs[4], width=8, color='m', align='center', label=list_name_of_senders[4])
    ax.set_xlabel("Time in Month and Year")
    ax.set_ylabel("# of Unique emails received")
    ax.legend()
    ax.xaxis_date()
    figure.savefig('Unique emails received by top sender over time', dpi=figure.dpi)
    