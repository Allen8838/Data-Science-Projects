"""
create histograms
"""

import matplotlib.pyplot as plt

def create_histograms(tst_histo, ctrl_histo):
    plt.hist(tst_histo, 30, label='Test group', facecolor='green')
    plt.hist(ctrl_histo, 30, label='Control group', facecolor='orange')
    plt.legend(loc='upper right')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Number of instances')
    plt.title(r'Probability distributions of test and control group')
    plt.savefig('Probability_distributions_of_test_and_control_group.png')
