from collections import defaultdict 
from itertools import repeat

test_group_list = []
trans_amt_test_group_list = []

control_group_list = []
trans_amt_control_group_list = []

list_test_for_histogram = []
list_control_for_histogram = []

control_dictionary = defaultdict(int)
test_dictionary = defaultdict(int)

create_test_list = lambda x: test_group_list.append(x) if x==1 else None
create_control_list = lambda x: control_group_list.append(x) if x==0 else None

create_trans_test_list = lambda x: trans_amt_test_group_list.append(x) if int(x) != 0.00 else None
create_trans_control_list = lambda x: trans_amt_control_group_list.append(x) if int(x) != 0.00 else None 

#creating this to add to dataframe. this will then be multiplied against the trnasaction amount so that we can get a column of only transaction amounts associated with control groups
indicator_column_for_control = lambda x: int(x)+1 if int(x) == 0 else 0


def get_lists_for_test_control(dataframe, test_group, transaction_amount):
    
    
    dataframe[test_group].map(create_test_list)
    dataframe[test_group].map(create_control_list)

    #create two additional columns in the dataframe, based on whether the transaction amount is associated with test or control
    dataframe['trans_test_group'] = dataframe[test_group]*dataframe[transaction_amount]
    dataframe['one_control_zero_test_indicator_column'] = dataframe[test_group].map(indicator_column_for_control)
    dataframe['trans_control_group'] = dataframe['one_control_zero_test_indicator_column']*dataframe[transaction_amount]

    dataframe['trans_test_group'].map(create_trans_test_list)
    dataframe['trans_control_group'].map(create_trans_control_list)
    
    list_tuple_of_test_group_trans_amt = zip(test_group_list, trans_amt_test_group_list)
    list_tuple_of_control_group_trans_amt = zip(control_group_list, trans_amt_control_group_list)
    
    master_list_tuple = list(list_tuple_of_test_group_trans_amt) + list(list_tuple_of_control_group_trans_amt)

    for tuple_pair in master_list_tuple:
        if int(tuple_pair[0]) == 1:
            test_dictionary[tuple_pair[1]] += 1
        else:
            control_dictionary[tuple_pair[1]] += 1

    for key, value in test_dictionary.items():
        list_test_for_histogram.extend(repeat(key, value))

    for key, value in control_dictionary.items():
        list_control_for_histogram.extend(repeat(key, value))
    

    return list_test_for_histogram, list_control_for_histogram










