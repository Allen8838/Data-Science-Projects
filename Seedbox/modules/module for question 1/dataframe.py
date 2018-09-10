import pandas as pd 
import numpy as np

def group_dataframe_column_n_user_type(dataframe, column_to_group, test_group):
    #this creates a groupby object. see line dataframe_only_REBILL.append(dataframe_grouped.get_group(key), ignore_index = True)
    #that allows access of items in the groupby object and split the groupby object
    dataframe_grouped = dataframe.groupby(column_to_group)

    dataframe_only_REBILL = pd.DataFrame()
    dataframe_REBILL_control_group = pd.DataFrame()
    dataframe_REBILL_test_group = pd.DataFrame()

    #create a dataframe where the transaction type is only REBILL
    #need to put it in key, _ format. if only loop as key in dataframe_grouped, then key is an entire dataframe
    for key, _ in dataframe_grouped:
        if key == 'REBILL':
            #both the for loop and append statement unpacks the dataframe object and allows the entire dataframe to be seen if printed
            dataframe_only_REBILL = dataframe_only_REBILL.append(dataframe_grouped.get_group(key), ignore_index = True)
    
    #bifurcate the dataframe_only_REBILL into control and test group by repeating similar process above
    dataframe_only_REBILL_split_users = dataframe_only_REBILL.groupby(test_group)

    #split off dataframe with REBILL by user type and assign to the empty dataframes defined above to unpack them
    for key, _ in dataframe_only_REBILL_split_users:
        if key == 1:
            dataframe_REBILL_test_group = dataframe_REBILL_test_group.append(dataframe_only_REBILL_split_users.get_group(key), ignore_index=True)
        if key == 0:
            dataframe_REBILL_control_group = dataframe_REBILL_control_group.append(dataframe_only_REBILL_split_users.get_group(key), ignore_index=True)
    
    return dataframe_REBILL_test_group, dataframe_REBILL_control_group


def group_test_control_by_sample_id_n_column(dataframe_REBILL_test_group, dataframe_REBILL_control_group, sample_id, column_to_group):
    #delete the transaction_id column from both dataframes as each transaction_id is unique and will prevent grouping by sample id later one
    dataframe_REBILL_test_group = dataframe_REBILL_test_group.drop(['transaction_id'], axis=1)
    dataframe_REBILL_control_group = dataframe_REBILL_control_group.drop(['transaction_id'], axis=1)

    #delete the transaction_amount as well as this information will be lost anyway once we groupby and count
    dataframe_REBILL_test_group = dataframe_REBILL_test_group.drop(['transaction_amount'], axis=1)
    dataframe_REBILL_control_group = dataframe_REBILL_control_group.drop(['transaction_amount'], axis=1)
    

    dataframe_test_groupby_sample_id = dataframe_REBILL_test_group.groupby([sample_id, column_to_group]).agg(['count'])
    dataframe_control_groupby_sample_id = dataframe_REBILL_control_group.groupby([sample_id, column_to_group]).agg(['count'])

    return dataframe_test_groupby_sample_id, dataframe_control_groupby_sample_id




def preprocess_dataframe(df1, df2, sample_id):
    combined_dataframe = pd.merge(df1, df2, on=sample_id)
    
    #go through all columns and replace empty cells with nan, so that we can use dropna later and delete empty rows
    combined_dataframe.replace(r'^\s*$', np.nan, regex=True, inplace = True)

    combined_dataframe.dropna(inplace=True)

    return combined_dataframe


