import matplotlib.pyplot as plt



def create_histograms(list_test_for_histogram, list_control_for_histogram):
    plt.hist(list_test_for_histogram, 40, label='Test group', facecolor='green')
    plt.hist(list_control_for_histogram, 40, label= 'Control group', facecolor='orange')
    plt.legend(loc='upper right')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Number of instances')
    plt.title(r'Probability distributions of test and control group')
    plt.show()
