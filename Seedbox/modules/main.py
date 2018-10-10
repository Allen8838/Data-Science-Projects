"""
main module to run submodules
"""

import pandas as pd

from dataframe import jn_df_n_rmv_empty_rws,\
                      split_df_usr_group_trans,\
                      group_smpl_id_n_col,\
                      state_cols_keep
from scipy import stats
from data_prep import get_ls_trans_amts_tst_con
from plot import create_histograms
from statistics import calc_tval_n_degfree


if __name__ == "__main__":
    """
    Question 1
    """

    #read in datasets
    df_test_samples = pd.read_csv(r'testSamples.csv')
    df_trans_data = pd.read_csv(r'transData.csv')

    #join the two dataframes together and delete empty rows
    df_prcsd = jn_df_n_rmv_empty_rws(df_test_samples, df_trans_data, 'sample_id')

    #creating another variable for df_prcsd as manipulations will need
    #to be performed on it for histogram but want to keep original dataframe
    #processed for subsequent tasks
    df_prcsd_for_histo = df_prcsd

    #plot the histograms
    tst_histo, ctrl_histo = get_ls_trans_amts_tst_con(df_prcsd_for_histo,
                                                      'test_group',
                                                      'transaction_amount')

    #since we have some negative values, we cannot pass 
    #tst_histo, ctrl_histo directly into the chi2_contingency table
    #we will square each element in the list first and pass the
    #resulting lists into the contigency function
    tst_histo_sq = [element**2 for element in tst_histo]
    ctrl_histo_sq = [element**2 for element in ctrl_histo]

    chi2_stat_test, p_val_test, dof_test, ex_test = stats.chi2_contingency(tst_histo_sq)

    print("===Chi2 Stat of Test Set===")
    print(chi2_stat_test)
    print("\n")
    print("===Degrees of Freedom===")
    print(dof_test)
    print("\n")
    print("===P-Value===")
    print(p_val_test)
    print("\n")
    print("===Contingency Table===")
    print(ex_test)

    chi2_stat_train, p_val_train, dof_train, ex_train = stats.chi2_contingency(ctrl_histo_sq)

    print("===Chi2 Stat of Control Set===")
    print(chi2_stat_train)
    print("\n")
    print("===Degrees of Freedom===")
    print(dof_train)
    print("\n")
    print("===P-Value===")
    print(p_val_train)
    print("\n")
    print("===Contingency Table===")
    print(ex_train)

    
    create_histograms(tst_histo, ctrl_histo)


    # """
    # Question 2
    # """
    # REBILL_tst_grp, REBILL_ctrl_grp = split_df_usr_group_trans(df=df_prcsd,
    #                                                            column_to_bifurcate='transaction_type',
    #                                                            user_group_to_bifurcate='test_group',
    #                                                            subtrans_typ_split='REBILL',
    #                                                            col_drop1=None, col_drop2=None)

    # #further prune down the dataframe to only the columns we want
    # REBILL_tst_grp = state_cols_keep(REBILL_tst_grp, ['sample_id',
    #                                                   'test_group',
    #                                                   'transaction_type'])

    # REBILL_ctrl_grp = state_cols_keep(REBILL_ctrl_grp, ['sample_id',
    #                                                     'test_group',
    #                                                     'transaction_type'])

    # tst_groupby_smpl_id, ctrl_groupby_smpl_id = group_smpl_id_n_col(REBILL_tst_grp, REBILL_ctrl_grp,
    #                                                                   'sample_id', 'transaction_type')

    # t_value_REBILL, degrees_of_freedom_REBILL = calc_tval_n_degfree(tst_groupby_smpl_id, ctrl_groupby_smpl_id)

    # """
    # Question 3
    # """
    # trans_amt_tst, trans_amt_ctrl = split_df_usr_group_trans(df=df_prcsd,
    #                                                          column_to_bifurcate='sample_id',
    #                                                          user_group_to_bifurcate='test_group',
    #                                                          subtrans_typ_split=None,
    #                                                          col_drop1='transaction_id',
    #                                                          col_drop2='transaction_type')

    # trans_amt_tst = state_cols_keep(trans_amt_tst, ['sample_id',
    #                                                 'test_group',
    #                                                 'transaction_amount'])

    # trans_amt_ctrl = state_cols_keep(trans_amt_ctrl, ['sample_id',
    #                                                   'test_group',
    #                                                   'transaction_amount'])

    # trans_amt_tst, trans_amt_ctrl = group_smpl_id_n_col(trans_amt_tst,
    #                                                       trans_amt_ctrl,
    #                                                       'sample_id',
    #                                                       'transaction_amount')

    # #groupby and aggregation turns columns into index. need to reset the index to make these columns accessible again
    # trans_amt_tst.reset_index(inplace=True)
    # trans_amt_ctrl.reset_index(inplace=True)

    # #check the shape of the dataframe to make sure they are of the right size. needed to reshape transaction amount column
    # trans_amt_tst['total_transaction_amount'] = trans_amt_tst['transaction_amount'].values.reshape(-1, 1)*trans_amt_tst['test_group']
    # trans_amt_ctrl['total_transaction_amount'] = trans_amt_ctrl['transaction_amount'].values.reshape(-1, 1)*trans_amt_ctrl['test_group']

    # #remove columns not needed to calculate t value
    # trans_amt_tst = state_cols_keep(trans_amt_tst, ['total_transaction_amount'])
    # trans_amt_ctrl = state_cols_keep(trans_amt_ctrl, ['total_transaction_amount'])

    # t_value_trans_amt, degrees_of_freedom_trans_amt = calc_tval_n_degfree(trans_amt_tst, trans_amt_ctrl)

    # """
    # Question 4
    # """
    # #Note that we can use variables: REBILL_tst_grp, REBILL_ctrl_grp defined above,
    # #get something similar for CHARGEBACK and then merge the resulting dataframes,
    # #based on user group
    # CHRGBCK_tst, CHRGBCK_ctrl = split_df_usr_group_trans(df=df_prcsd,
    #                                                      column_to_bifurcate='transaction_type',
    #                                                      user_group_to_bifurcate='test_group',
    #                                                      subtrans_typ_split='CHARGEBACK',
    #                                                      col_drop1='transaction_id',
    #                                                      col_drop2='transaction_amount')

    # #get a count by transaction type grouping on sample_id, for each user group
    # CHRGBCK_tst_smpl_id, CHRGBCK_ctrl_smpl_id = group_smpl_id_n_col(CHRGBCK_tst,
    #                                                                   CHRGBCK_ctrl,
    #                                                                   'sample_id',
    #                                                                   'transaction_type')

    # #reset indexes on dataframes to make columns accessible
    # CHRGBCK_tst_smpl_id.reset_index(inplace=True)
    # CHRGBCK_ctrl_smpl_id.reset_index(inplace=True)
    # tst_groupby_smpl_id.reset_index(inplace=True)
    # ctrl_groupby_smpl_id.reset_index(inplace=True)

    # #merge horizontally the test dataframes and the control dataframes
    # CHRGBCK_REBILL_tst = pd.merge(CHRGBCK_tst_smpl_id, tst_groupby_smpl_id, on='sample_id')
    # CHRGBCK_REBILL_ctrl = pd.merge(CHRGBCK_ctrl_smpl_id, ctrl_groupby_smpl_id, on='sample_id')

    # #flatten the dataframe columns as we have two rows of headers. then rename the columns
    # CHRGBCK_REBILL_tst.columns = [f'{i}|{j}' if j != '' else f'{i}' for i, j in CHRGBCK_REBILL_tst.columns]
    # CHRGBCK_REBILL_ctrl.columns = [f'{i}|{j}' if j != '' else f'{i}' for i, j in CHRGBCK_REBILL_ctrl.columns]

    # CHRGBCK_REBILL_tst = CHRGBCK_REBILL_tst.rename(index=str, columns={'test_group_x|count':'count of chargebacks',
    #                                                                    'test_group_y|count':'count of rebills'})

    # CHRGBCK_REBILL_ctrl = CHRGBCK_REBILL_ctrl.rename(index=str, columns={'test_group_x|count':'count of chargebacks',
    #                                                                      'test_group_y|count':'count of rebills'})

    # #create a column to calculate chargeback/rebill
    # CHRGBCK_REBILL_tst['CHARGEBACKS/REBILLS'] = CHRGBCK_REBILL_tst['count of chargebacks']/CHRGBCK_REBILL_tst['count of rebills']
    # CHRGBCK_REBILL_ctrl['CHARGEBACKS/REBILLS'] = CHRGBCK_REBILL_ctrl['count of chargebacks']/CHRGBCK_REBILL_ctrl['count of rebills']

    # #remove columns no longer needed
    # CHRGBCK_REBILL_tst = state_cols_keep(CHRGBCK_REBILL_tst, ['CHARGEBACKS/REBILLS'])
    # CHRGBCK_REBILL_ctrl = state_cols_keep(CHRGBCK_REBILL_ctrl, ['CHARGEBACKS/REBILLS'])

    # #calculate t value and degrees of freedom
    # tval_chrgbck_rt, degfree_chrgbck_rt = calc_tval_n_degfree(CHRGBCK_REBILL_tst, CHRGBCK_REBILL_ctrl)
    