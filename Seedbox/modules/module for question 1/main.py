from data_for_histograms import get_lists_for_test_control
from plot import create_histograms 
from statistics import calculate_t_value_n_degrees_of_freedom
from dataframe import preprocess_dataframe, group_dataframe_column_n_user_type, group_test_control_by_sample_id_n_column
import pandas as pd




if __name__ == "__main__":
    dataframe_test_samples = pd.read_csv(r'testSamples.csv')
    dataframe_trans_data = pd.read_csv(r'transData.csv')
    dataframe_processed = preprocess_dataframe(dataframe_test_samples, dataframe_trans_data, 'sample_id')
    dataframe_REBILL_test_group, dataframe_REBILL_control_group = group_dataframe_column_n_user_type(dataframe_processed, 'transaction_type', 'test_group')
    dataframe_test_groupby_sample_id, dataframe_control_groupby_sample_id = group_test_control_by_sample_id_n_column(dataframe_REBILL_test_group, dataframe_REBILL_control_group, 'sample_id', 'transaction_type')
    t_value_REBILL, degrees_of_freedom_REBILL = calculate_t_value_n_degrees_of_freedom(dataframe_test_groupby_sample_id, dataframe_control_groupby_sample_id, 'sample_id')
    list_test_for_histogram, list_control_for_histogram = get_lists_for_test_control(dataframe_processed, 'test_group', 'transaction_amount')
    create_histograms(list_test_for_histogram, list_control_for_histogram)

    
