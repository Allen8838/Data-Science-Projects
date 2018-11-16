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

rowsums = train.iloc[:,2:].sum(axis=1)
# rowsums adds up each column and returns the row number and the sum value, like so
# 0         0
# 1         0
# 2         0
# 3         0
# 4         0
# with length of 159571

train['clean'] = (rowsums==0)

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


temp_df = train.iloc[:,2:-1]
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


# feature engineering

# indirect features
merge = pd.concat([train.iloc[:,0:2], test.iloc[:,0:2]])
df = merge.reset_index(drop=True)

df['count_sent'] = df['comment_text'].apply(lambda x: len(re.findall("\n", str(x)))+1)
df['count_word'] = df['comment_text'].apply(lambda x: len(str(x).split()))
df['count_unique_word'] = df['comment_text'].apply(lambda x: len(set(str(x).split())))
df['count_letters'] = df['comment_text'].apply(lambda x: len(str(x)))
df['count_punctuations'] = df['comment_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
df['count_words_upper'] = df['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
df['count_words_title'] = df['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
df['count_stopwords'] = df['comment_text'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
df['mean_word_len'] = df['comment_text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# derived features
df['word_unique_percent'] = df['count_unique_word']*100/df['count_word']
df['punct_percent'] = df['count_punctuations']*100/df['count_word']

train_feats = df.iloc[0:len(train),]
# train_feats will look like this
#                       id                                       comment_text  count_sent  count_word      ...        count_stopwords  mean_word_len  word_unique_percent  punct_percent
# 0       0000997932d777bf  Explanation\nWhy the edits made under my usern...           2          43      ...                     18       5.162791            95.348837      23.255814
# 1       000103f0d9cfb60f  D'aww! He matches this background colour I'm s...           1          17      ...                      2       5.588235           100.000000      70.588235
# with a final shape of 159,571 x 13 columns

test_feats = df.iloc[len(train):,]
# test_feats will look like 
#                       id                                       comment_text  count_sent  count_word      ...        count_stopwords  mean_word_len  word_unique_percent  punct_percent
# 159571  00001cee341fdb12  Yo bitch Ja Rule is more succesful then you'll...           1          72      ...                     26       4.111111            84.722222      16.666667
# 159572  0000247867823ef7  == From RfC == \n\n The title is fine as it is...           3          12      ...                      5       3.000000            91.666667      50.000000
# with a final shape of 153,164 x 13 columns

train_tags = train.iloc[:,2:]
# train_tags will look like this
#         toxic  severe_toxic  obscene  threat  insult  identity_hate
# 0           0             0        0       0       0              0
# 1           0             0        0       0       0              0
# with a shape of 159,571 x 6 columns

train_feats = pd.concat([train_feats, train_tags], axis=1)
# train_feats will look like this
#                       id                                       comment_text  count_sent  count_word  count_unique_word      ...        severe_toxic  obscene  threat  insult  identity_hate
# 0       0000997932d777bf  Explanation\nWhy the edits made under my usern...           2          43                 41      ...                   0        0       0       0              0
# 1       000103f0d9cfb60f  D'aww! He matches this background colour I'm s...           1          17                 17      ...                   0        0       0       0              0
# with a shape of 159,571 x 19 columns

# Are longer comments more toxic?
train_feats['count_sent'].loc[train_feats['count_sent']>10] = 10
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.suptitle("Are longer comments more toxic?", fontsize=20)
sns.violinplot(y='count_sent', x='clean', data=train_feats, split=True)
plt.xlabel('Clean?', fontsize=12)
plt.ylabel('# of sentences', fontsize=12)
plt.title('Number of sentences in each comment', fontsize=15)
train_feats['count_word'].loc[train_feats['count_word']>200] = 200
plt.subplot(122)
sns.violinplot(y='count_word', x='clean', data=train_feats, split=True, inner='quart')
plt.xlabel('Clean?', fontsize=12)
plt.ylabel('# of words', fontsize=12)
plt.title('Number of words in each comment', fontsize=15)
plt.savefig('Are longer comments more toxic.png')

train_feats['count_unique_word'].loc[train_feats['count_unique_word']>200] = 200

temp_df = pd.melt(train_feats, value_vars=['count_word', 'count_unique_word'], id_vars='clean')

spammers = train_feats[train_feats['word_unique_percent']<30]

plt.figure(figsize=(16, 12))
plt.suptitle("What's so unique?", fontsize=20)
gridspec.GridSpec(2,2)
plt.subplot2grid((2,2), (0,0))
sns.violinplot(x='variable', y='value', hue='clean', data=temp_df, split=True, inner='quartile')
plt.title("Absolute wordcount and unique word count")
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.subplot2grid((2,2), (0,1))
plt.title("Percentage of unique words of total words in comment")
ax = sns.kdeplot(train_feats[train_feats.clean==0].word_unique_percent, label='Bad', shade=True, color='r')
ax = sns.kdeplot(train_feats[train_feats.clean==1].word_unique_percent, label='Clean')
plt.legend()
plt.ylabel('Number of occurences', fontsize=12)
plt.xlabel('Percent of unique words', fontsize=12)

x = spammers.iloc[:,-7:].sum()
plt.subplot2grid((2,2), (1,0), colspan=2)
plt.title('Count of comments with low(<30%) unique words', fontsize=15)
ax = sns.barplot(x=x.index, y=x.values, color=color[3])

rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+5, label, ha='center', va='bottom')

plt.xlabel('Threat class', fontsize=12)
plt.ylabel('# of comments', fontsize=12)
plt.savefig('Uniqueness.png')

