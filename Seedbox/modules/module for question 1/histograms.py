#general overview for code below is:
#find the transaction amounts associated with the test group, put it into a list and return it
#find the transaction amounts associated with the control group, put it into a list and return it 

from collections import defaultdict 
from itertools import repeat

test_group_list = []
trans_amt_test_group_list = []

control_group_list = []
trans_amt_control_group_list = []

create_trans_test_list = lambda x: trans_amt_test_group_list.append(x) if int(x) != 0.00 else None
create_trans_control_list = lambda x: trans_amt_control_group_list.append(x) if int(x) != 0.00 else None 

#creating this to add to dataframe. this will then be multiplied against the trnasaction amount so that we can get a column of only transaction amounts associated with control groups
indicator_column_for_control = lambda x: int(x)+1 if int(x) == 0 else 0


def get_lists_trans_amts_for_test_n_control(dataframe, user_group, transaction_amount):
    #create two columns in the dataframe, based on whether the transaction amount is associated with test or control
    #since our user group is only 1's and 0's, the code just below will only select the test group with a transaction amount
    dataframe['trans_test_group'] = dataframe[user_group]*dataframe[transaction_amount] 
    #creating an indicator column to flip the 1's and 0's, so that 1 indicates the presence of a control group
    dataframe['one_control_zero_test_indicator_column'] = dataframe[user_group].map(indicator_column_for_control)
    dataframe['trans_control_group'] = dataframe['one_control_zero_test_indicator_column']*dataframe[transaction_amount]

    dataframe['trans_test_group'].map(create_trans_test_list)
    dataframe['trans_control_group'].map(create_trans_control_list)
    
   
    #reset the dataframe to the way it was so that it doesn't interfere with subsequent analyses
    dataframe = dataframe

    return trans_amt_test_group_list, trans_amt_control_group_list 









