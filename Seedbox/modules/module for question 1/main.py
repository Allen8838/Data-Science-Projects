from preprocess_dataframe import preprocess_dataframe
from data_for_histograms import get_lists_for_test_control
from plot import create_histograms 
import pandas as pd




if __name__ == "__main__":
    dataframe_test_samples = pd.read_csv(r'testSamples.csv')
    dataframe_trans_data = pd.read_csv(r'transData.csv')
    dataframe_processed = preprocess_dataframe(dataframe_test_samples, dataframe_trans_data, 'sample_id')
    list_test_for_histogram, list_control_for_histogram = get_lists_for_test_control(dataframe_processed, 'test_group', 'transaction_amount')
    create_histograms(list_test_for_histogram, list_control_for_histogram)

    
