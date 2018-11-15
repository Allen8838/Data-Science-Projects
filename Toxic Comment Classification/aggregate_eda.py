# from https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda

import numpy as np
import pandas as pd

import gc
import time
import warnings

# stats
from scipy.misc import imread
from scipy import sparse
import scipy.stats as ss

# visualizations
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import matplotlib_venn as venn

# nlp
import string
import re
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

# Feature Engineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# settings
start_time = time.time()
color = sns.color_palette()
sns.set_style("dark")
eng_stopwords = set(stopwords.words("english"))
warnings.filterwarnings("ignore")

lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()

# read in the datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# class imbalance
print("Check for missing values in Train dataset")
null_check = train.isnull().sum()
print(null_check)
print("Check for missing values in Test dataset")
null_check = test.isnull().sum()
print(null_check)

# fill NA with unknown
train['comment_text'].fillna('unknown', inplace=True)
test['comment_text'].fillna('unknown', inplace=True)


temp_df = train.iloc[:,2:0]
# this is what temp_df will look like
#           toxic  severe_toxic  obscene  threat  insult   identity_hate
# 0           0             0        0       0       0           0
# 1           0             0        0       0       0           0
# the 2 gets rid of the first two columns which are id and comment_text

main_col = 'toxic'
corr_mats = []

for other_col in temp_df.columns[1:]:
    confusion_matrix = pd.crosstab(temp_df[main_col], temp_df[other_col])
    corr_mats.append(confusion_matrix)

out = pd.concat(corr_mats, axis=1, keys=temp_df.columns[1:])


def cramers_corrected_stat(confusion_matrix):
    """
    calculate Cramers V statistic for categorical-categorical association
    uses correction from Bergsma and Wicher
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))

col1 = "toxic"
col2 = "severe_toxic"

confusion_matrix = pd.crosstab(temp_df[col1], temp_df[col2])
# the confusion matrix will look like the following
# severe_toxic           0             1
# toxic 
#   0                  144277          0
#   1                   13699       1595

new_corr = cramers_corrected_stat(confusion_matrix)
