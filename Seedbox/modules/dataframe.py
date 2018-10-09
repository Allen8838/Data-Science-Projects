"""
module for dataframe processing
"""

import pandas as pd
import numpy as np

def split_df_usr_group_trans(df=None,
                             column_to_bifurcate=None,
                             user_group_to_bifurcate=None,
                             subtrans_typ_split=None,
                             col_drop1=None, col_drop2=None):
    """
    splits the dataframe user group by transaction amount
    and returns a test group and control group
    """
    #delete any columns specified as unnecessary
    if col_drop1 != None:
        df = df.drop([col_drop1], axis=1)

    if col_drop2 != None:
        df = df.drop([col_drop2], axis=1)

    #this creates a groupby object. see line
    #sub_df.append(df_grouped.get_group(key), ignore_index = True)
    #that allows access of items in the groupby object and split the groupby object
    df_grouped = df.groupby(column_to_bifurcate)

    sub_df = pd.DataFrame()
    df_ctrl_grp = pd.DataFrame()
    df_tst_grp = pd.DataFrame()

    #need to put it in key, _ format. if only loop as key in
    #df_grouped, then key is an entire df
    if subtrans_typ_split != None:
        for key, _ in df_grouped:
            if key == subtrans_typ_split:
                #both the for loop and append statement unpacks the df object
                #and allows the entire df to be seen if printed
                sub_df = sub_df.append(df_grouped.get_group(key), ignore_index=True)
    else:
        for key, _ in df_grouped:
            sub_df = sub_df.append(df_grouped.get_group(key), ignore_index=True)

    #bifurcate the sub_df into control and test group by repeating similar process above
    df_split_users = sub_df.groupby(user_group_to_bifurcate)

    #split off df by user type and assign to the empty dfs defined above to unpack them
    for key, _ in df_split_users:
        if key == 1:
            df_tst_grp = df_tst_grp.append(df_split_users.get_group(key), ignore_index=True)
        if key == 0:
            df_ctrl_grp = df_ctrl_grp.append(df_split_users.get_group(key), ignore_index=True)

    return df_tst_grp, df_ctrl_grp


def state_cols_keep(df, cols_keep):
    df = df[cols_keep]

    return df



def group_smpl_id_n_col(df_tst_grp, df_ctrl_grp, smpl_id, col_grp):
    """
    group by the sample id and the column specified by the user
    """
    try:
        #delete the following columns if not already deleted
        df_tst_grp = df_tst_grp.drop(['transaction_id'], axis=1)
        df_ctrl_grp = df_ctrl_grp.drop(['transaction_id'], axis=1)

    except:
        pass

    df_tst_smpl_id = df_tst_grp.groupby([smpl_id, col_grp]).agg(['count'])
    df_ctrl_smpl_id = df_ctrl_grp.groupby([smpl_id, col_grp]).agg(['count'])

    return df_tst_smpl_id, df_ctrl_smpl_id



def jn_df_n_rmv_empty_rws(df1, df2, smpl_id):
    """
    merge two dataframes together by sample id
    """
    combined_df = pd.merge(df1, df2, on=smpl_id)

    #go through all columns and replace empty cells with nan, so that we can use
    #dropna later and delete empty rows
    combined_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    combined_df.dropna(inplace=True)

    return combined_df
