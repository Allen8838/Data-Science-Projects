import pandas as pd 

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
            dataframe_only_REBILL = dataframe_only_REBILL.append(dataframe_grouped.get_group(key), ignore_index = True)
    
    #bifurcate the dataframe_only_REBILL into control and test group by repeating similar process above
    dataframe_only_REBILL_split_users = dataframe_only_REBILL.groupby(test_group)

    #split off dataframe with REBILL by user type and assign to the empty dataframes defined above
    for key, _ in dataframe_only_REBILL_split_users:
        if key == 1:
            dataframe_REBILL_test_group = dataframe_REBILL_test_group.append(dataframe_only_REBILL_split_users.get_group(key), ignore_index = True)
        if key == 0:
            dataframe_REBILL_control_group = dataframe_REBILL_control_group.append(dataframe_only_REBILL_split_users.get_group(key), ignore_index = True)
    
    return dataframe_REBILL_test_group, dataframe_REBILL_control_group


def group_test_control_by_sample_id(dataframe_REBILL_test_group, dataframe_REBILL_control_group, sample_id):
    
    dataframe_test_groupby_sample_id = pd.DataFrame()
    dataframe_control_groupby_sample_id = pd.DataFrame()



