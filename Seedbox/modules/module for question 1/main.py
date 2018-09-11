from data_for_histograms import get_lists_for_test_control
from plot import create_histograms 
from statistics import calculate_t_value_n_degrees_of_freedom
from dataframe import preprocess_dataframe, group_dataframe_column_n_user_type, group_test_control_by_sample_id_n_column, state_columns_to_keep
import pandas as pd




if __name__ == "__main__":
    #QUESTION 1
    dataframe_test_samples = pd.read_csv(r'testSamples.csv')
    dataframe_trans_data = pd.read_csv(r'transData.csv')
    #join the two dataframes together and delete empty rows
    dataframe_processed = preprocess_dataframe(dataframe_test_samples, dataframe_trans_data, 'sample_id')

    #QUESTION 2
    dataframe_REBILL_test_group, dataframe_REBILL_control_group = group_dataframe_column_n_user_type(dataframe=dataframe_processed, column_to_group='transaction_type', user_group='test_group', subtransaction_type='REBILL', column_to_drop1=None, column_to_drop2=None)

    #further prune down the dataframe to only the columns we want
    dataframe_REBILL_test_group = state_columns_to_keep(dataframe_REBILL_test_group, ['sample_id', 'test_group', 'transaction_type'])  
    dataframe_REBILL_control_group = state_columns_to_keep(dataframe_REBILL_control_group, ['sample_id', 'test_group', 'transaction_type'])  

    dataframe_test_groupby_sample_id, dataframe_control_groupby_sample_id = group_test_control_by_sample_id_n_column(dataframe_REBILL_test_group, dataframe_REBILL_control_group, 'sample_id', 'transaction_type')

    t_value_REBILL, degrees_of_freedom_REBILL = calculate_t_value_n_degrees_of_freedom(dataframe_test_groupby_sample_id, dataframe_control_groupby_sample_id)

    #QUESTION 3 
    dataframe_trans_amt_test_group, dataframe_trans_amt_control_group = group_dataframe_column_n_user_type(dataframe=dataframe_processed, column_to_group='sample_id', user_group='test_group', subtransaction_type=None, column_to_drop1='transaction_id', column_to_drop2='transaction_type')
    
    dataframe_trans_amt_test_group = state_columns_to_keep(dataframe_trans_amt_test_group, ['sample_id', 'test_group', 'transaction_amount'])
    dataframe_trans_amt_control_group = state_columns_to_keep(dataframe_trans_amt_control_group, ['sample_id', 'test_group', 'transaction_amount'])

    dataframe_trans_amt_test, dataframe_trans_amt_control = group_test_control_by_sample_id_n_column(dataframe_trans_amt_test_group, dataframe_trans_amt_control_group, 'sample_id', 'transaction_amount')

    #groupby and aggregation turns columns into index. need to reset the index to make these columns accessible again
    dataframe_trans_amt_test.reset_index(inplace=True)
    dataframe_trans_amt_control.reset_index(inplace=True)

    #check the shape of the dataframe to make sure they are of the right size. needed to reshape transaction amount column
    dataframe_trans_amt_test['total_transaction_amount'] = dataframe_trans_amt_test['transaction_amount'].values.reshape(-1,1)*dataframe_trans_amt_test['test_group']
    dataframe_trans_amt_control['total_transaction_amount'] =  dataframe_trans_amt_control['transaction_amount'].values.reshape(-1,1)*dataframe_trans_amt_control['test_group']
    
    #remove columns not needed to calculate t value
    dataframe_trans_amt_test = state_columns_to_keep(dataframe_trans_amt_test, ['total_transaction_amount'])
    dataframe_trans_amt_control = state_columns_to_keep(dataframe_trans_amt_control, ['total_transaction_amount'])

    t_value_trans_amt, degrees_of_freedom_trans_amt = calculate_t_value_n_degrees_of_freedom(dataframe_trans_amt_test, dataframe_trans_amt_control)

    #print(dataframe_trans_amt_test)
    #QUESTION 4
    #Note that we can use variables: dataframe_REBILL_test_group, dataframe_REBILL_control_group defined above, get something similar for CHARGEBACK
    #and then merge the resulting dataframes, based on user group
    dataframe_CHARGEBACK_test_group, dataframe_CHARGEBACK_control_group = group_dataframe_column_n_user_type(dataframe=dataframe_processed, column_to_group='transaction_type', user_group='test_group', subtransaction_type='CHARGEBACK', column_to_drop1='transaction_id', column_to_drop2='transaction_amount')
    
    #get a count by transaction type grouping on sample_id, for each user group
    dataframe_CHARGEBACK_test_groupby_sample_id, dataframe_CHARGEBACK_control_groupby_sample_id = group_test_control_by_sample_id_n_column(dataframe_CHARGEBACK_test_group, dataframe_CHARGEBACK_control_group, 'sample_id', 'transaction_type')
    
    #reset indexes on dataframes to make columns accessible
    dataframe_CHARGEBACK_test_groupby_sample_id.reset_index(inplace=True) 
    dataframe_CHARGEBACK_control_groupby_sample_id.reset_index(inplace=True)
    dataframe_test_groupby_sample_id.reset_index(inplace=True)
    dataframe_control_groupby_sample_id.reset_index(inplace=True)
    
    #merge horizontally the test dataframes and the control dataframes
    dataframe_CHARGEBACK_REBILL_test_group = pd.merge(dataframe_CHARGEBACK_test_groupby_sample_id, dataframe_test_groupby_sample_id, on='sample_id')
    dataframe_CHARGEBACK_REBILL_control_group = pd.merge(dataframe_CHARGEBACK_control_groupby_sample_id, dataframe_control_groupby_sample_id, on='sample_id')
    
    #flatten the dataframe columns as we have two rows of headers. then rename the columns
    dataframe_CHARGEBACK_REBILL_test_group.columns = [f'{i}|{j}' if j != '' else f'{i}' for i,j in dataframe_CHARGEBACK_REBILL_test_group.columns]
    dataframe_CHARGEBACK_REBILL_control_group.columns = [f'{i}|{j}' if j != '' else f'{i}' for i,j in dataframe_CHARGEBACK_REBILL_control_group.columns]

    dataframe_CHARGEBACK_REBILL_test_group = dataframe_CHARGEBACK_REBILL_test_group.rename(index=str, columns={'test_group_x|count':'count of chargebacks', 'test_group_y|count':'count of rebills'})
    dataframe_CHARGEBACK_REBILL_control_group = dataframe_CHARGEBACK_REBILL_control_group.rename(index=str, columns={'test_group_x|count':'count of chargebacks', 'test_group_y|count':'count of rebills'})

    #create a column to calculate chargeback/rebill
    dataframe_CHARGEBACK_REBILL_test_group['CHARGEBACKS/REBILLS'] = dataframe_CHARGEBACK_REBILL_test_group['count of chargebacks']/dataframe_CHARGEBACK_REBILL_test_group['count of rebills']
    dataframe_CHARGEBACK_REBILL_control_group['CHARGEBACKS/REBILLS'] = dataframe_CHARGEBACK_REBILL_control_group['count of chargebacks']/dataframe_CHARGEBACK_REBILL_control_group['count of rebills']

    #remove columns no longer needed
    dataframe_CHARGEBACK_REBILL_test_group = state_columns_to_keep(dataframe_CHARGEBACK_REBILL_test_group, ['CHARGEBACKS/REBILLS'])
    dataframe_CHARGEBACK_REBILL_control_group = state_columns_to_keep(dataframe_CHARGEBACK_REBILL_control_group, ['CHARGEBACKS/REBILLS'])

    #calculate t value and degrees of freedom
    t_value_chargeback_rate, degrees_of_freedom_chargeback_rate = calculate_t_value_n_degrees_of_freedom(dataframe_CHARGEBACK_REBILL_test_group, dataframe_CHARGEBACK_REBILL_control_group)

    
    list_test_for_histogram, list_control_for_histogram = get_lists_for_test_control(dataframe_processed, 'test_group', 'transaction_amount')
    create_histograms(list_test_for_histogram, list_control_for_histogram)
