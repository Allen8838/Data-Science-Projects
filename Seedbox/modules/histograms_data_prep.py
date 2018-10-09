"""
general overview for code below is:
find the transaction amounts associated with the test group, put it into a list and return it
find the transaction amounts associated with the control group, put it into a list and return it
"""

def get_ls_trans_amts_tst_con(df, usr_grp, trans_amt):
    trans_amt_tst_grp = []
    trans_amt_ctrl_grp = []

    create_trans_tst = lambda x: trans_amt_tst_grp.append(x) if int(x) != 0.00 else None
    create_trans_ctrl = lambda x: trans_amt_ctrl_grp.append(x) if int(x) != 0.00 else None

    #creating this to add to df. this will then be multiplied against the
    #transaction amount so that we can get a column of only transaction amounts
    #associated with control groups
    indic_col_ctrl = lambda x: int(x)+1 if int(x) == 0 else 0

    #create two columns in the df, based on whether the transaction amount
    #is associated with test or control since our user group is only 1's and 0's, 
    #the code just below will only select the test group with a transaction amount
    df['trans_test_group'] = df[usr_grp]*df[trans_amt] 
    #creating an indicator column to flip the 1's and 0's, so that 1 indicates the presence of a control group
    df['one_control_zero_test_indicator_column'] = df[usr_grp].map(indic_col_ctrl)
    df['trans_control_group'] = df['one_control_zero_test_indicator_column']*df[trans_amt]

    df['trans_test_group'].map(create_trans_tst)
    df['trans_control_group'].map(create_trans_ctrl)
    
   
    #reset the df to the way it was so that it doesn't interfere with subsequent analyses
    df = df

    return trans_amt_tst_grp, trans_amt_ctrl_grp
