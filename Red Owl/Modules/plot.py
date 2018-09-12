import matplotlib.pyplot as plt
import time
import numpy as np
import datetime
import pylab 

def graph_top_senders(list_of_top_five_senders, dict_senders_number_msgs_per_time):
    number_of_subplots = len(list_of_top_five_senders)
    
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
            # date = num_messages_n_date[1]
            # year = date.split('-')[0]
            # month = date.split('-')[1]
            # day = date.split('-')[2]
            #gc.disable()
            date_array.append(datetime.datetime(int(year), int(month), 1, 0))
            number_messages_array.append(num_messages_n_date[0])
           # print(number_messages_array)
            axes = plt.subplot(number_of_subplots,1,i+1)
            axes.set_title('Emails sent over time by {}'.format(name_of_sender))
            axes.set_xlabel("Time in Month and Year")
            axes.set_ylabel("Number of Emails")
            axes.bar(date_array, number_messages_array, width=10)
            axes.xaxis_date()
            axes.set_anchor('W')
    #gc.enable()
    #plt.show()
    pylab.savefig('Number of emails sent over time by the top {} senders.png'.format(number_of_subplots))
    #plt.plot(date_array,number_messages_array)
    #plt.show()
    