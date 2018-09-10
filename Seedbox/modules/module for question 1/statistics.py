import math


def calculate_t_value_n_degrees_of_freedom(dataframe_test_groupby_sample_id, dataframe_control_groupby_sample_id, sample_id):
    standard_deviation_count_REBILL_test = dataframe_test_groupby_sample_id.std()
    standard_deviation_count_REBILL_control = dataframe_control_groupby_sample_id.std()

    mean_count_REBILL_test = dataframe_test_groupby_sample_id.mean()
    mean_count_REBILL_control = dataframe_control_groupby_sample_id.mean()

    #we want the number of unique person per user group, not the total number of rebills
    number_of_samples_REBILL_test = dataframe_test_groupby_sample_id.shape[0]
    number_of_samples_REBILL_control = dataframe_control_groupby_sample_id.shape[0]

    #null hypothesis is that the mean count REBILL of test group is less than or equal to the mean value of the control group. setting up the
    #null hypothesis this way will make it clearer if we do reject the null, that the mean count of test group is larger than the control group

    standard_error = math.sqrt(((standard_deviation_count_REBILL_test**2)/number_of_samples_REBILL_test)+((standard_deviation_count_REBILL_control**2)/number_of_samples_REBILL_control))
    t_value = (mean_count_REBILL_test- mean_count_REBILL_control)/standard_error

    degrees_of_freedom = number_of_samples_REBILL_test + number_of_samples_REBILL_control - 2

    return t_value, degrees_of_freedom