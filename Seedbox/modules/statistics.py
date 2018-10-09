"""
calculate t value and degrees of freedom
"""
import math


def calc_tval_n_degfree(tst_smpl_id, ctrl_smpl_id):
    std_dev_cnt_tst = tst_smpl_id.std()
    std_dev_cnt_ctrl = ctrl_smpl_id.std()

    mean_count_test = tst_smpl_id.mean()
    mean_count_control = ctrl_smpl_id.mean()

    #we want the number of unique person per user group, not the total number of rebills
    num_smpl_tst = tst_smpl_id.shape[0]
    num_smpl_ctrl = ctrl_smpl_id.shape[0]

    #null hypothesis is that the mean count REBILL of test group is less than or equal
    #to the mean value of the control group. setting up the null hypothesis this way will
    #make it clearer if we do reject the null, that the mean count of test group is larger
    #than the control group
    std_err = math.sqrt(((std_dev_cnt_tst**2)/num_smpl_tst)+((std_dev_cnt_ctrl**2)/num_smpl_ctrl))
    t_value = (mean_count_test- mean_count_control)/std_err

    degfree = num_smpl_tst + num_smpl_ctrl - 2

    return t_value, degfree
