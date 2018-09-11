import pandas as pd 
import numpy as np

def bifurcate_dataframe_by_user_group_n_trans(dataframe=None, column_to_bifurcate=None, user_group_to_bifurcate=None, subtrans_type_to_bifurcate=None, column_to_drop1=None, column_to_drop2=None):
    #delete any columns specified as unnecessary
    if column_to_drop1 != None:
        dataframe = dataframe.drop([column_to_drop1], axis=1)

    if column_to_drop2 != None:
        dataframe = dataframe.drop([column_to_drop2], axis=1)

    #this creates a groupby object. see line sub_dataframe.append(dataframe_grouped.get_group(key), ignore_index = True)
    #that allows access of items in the groupby object and split the groupby object
    dataframe_grouped = dataframe.groupby(column_to_bifurcate)

    sub_dataframe = pd.DataFrame()
    dataframe_control_group = pd.DataFrame()
    dataframe_test_group = pd.DataFrame()


    #need to put it in key, _ format. if only loop as key in dataframe_grouped, then key is an entire dataframe
    if subtrans_type_to_bifurcate != None:
        for key, _ in dataframe_grouped:
            if key == subtrans_type_to_bifurcate:
                #both the for loop and append statement unpacks the dataframe object and allows the entire dataframe to be seen if printed
                sub_dataframe = sub_dataframe.append(dataframe_grouped.get_group(key), ignore_index=True)
    else:
        for key, _ in dataframe_grouped:
            sub_dataframe = sub_dataframe.append(dataframe_grouped.get_group(key), ignore_index=True)

    #bifurcate the sub_dataframe into control and test group by repeating similar process above
    dataframe_split_users = sub_dataframe.groupby(user_group_to_bifurcate)

    #split off dataframe by user type and assign to the empty dataframes defined above to unpack them
    for key, _ in dataframe_split_users:
        if key == 1:
            dataframe_test_group = dataframe_test_group.append(dataframe_split_users.get_group(key), ignore_index=True)
        if key == 0:
            dataframe_control_group = dataframe_control_group.append(dataframe_split_users.get_group(key), ignore_index=True)
    
    return dataframe_test_group, dataframe_control_group


def state_columns_to_keep(dataframe, list_of_columns_to_keep):
    dataframe = dataframe[list_of_columns_to_keep]

    return dataframe



def group_test_control_by_sample_id_n_column(dataframe_test_group, dataframe_control_group, sample_id, column_to_group):
    try:
        #delete the following columns if not already deleted
        dataframe_test_group = dataframe_test_group.drop(['transaction_id'], axis=1)
        dataframe_control_group = dataframe_control_group.drop(['transaction_id'], axis=1)

    except:
        pass

    dataframe_test_groupby_sample_id = dataframe_test_group.groupby([sample_id, column_to_group]).agg(['count'])
    dataframe_control_groupby_sample_id = dataframe_control_group.groupby([sample_id, column_to_group]).agg(['count'])

    return dataframe_test_groupby_sample_id, dataframe_control_groupby_sample_id



def join_dataframes_n_drop_empty_rows(df1, df2, sample_id):
    combined_dataframe = pd.merge(df1, df2, on=sample_id)
    
    #go through all columns and replace empty cells with nan, so that we can use dropna later and delete empty rows
    combined_dataframe.replace(r'^\s*$', np.nan, regex=True, inplace = True)

    combined_dataframe.dropna(inplace=True)

    return combined_dataframe


