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
# merge will look like
# 0         0000997932d777bf      Explanation\nWhy the edits made under my usern...
# 1         000103f0d9cfb60f      D'aww! He matches this background colour I'm s...
# where the last row in the dataframe has an index of 153,163

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

# Leaky features
df['ip'] = df['comment_text'].apply(lambda x: re.findall('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', str(x)))

# count of ip addresses
df['count_ip'] = df['ip'].apply(lambda x: len(x))

df['link'] = df['comment_text'].apply(lambda x: re.findall("http://.*com", str(x)))

df["count_links"] = df['link'].apply(lambda x: len(x))

df['article_id'] = df['comment_text'].apply(lambda x: re.findall("\d:\d\d\s{0,5}$", str(x)))

df['article_id_flag'] = df.article_id.apply(lambda x: len(x))

df['username'] = df['comment_text'].apply(lambda x: re.findall("\[\[User(.*)\|", str(x)))

df['count_usernames'] = df['username'].apply(lambda x: len(x))

cv = CountVectorizer()
count_feats_ip = cv.fit_transform(df['ip'].apply(lambda x: str(x)))

leaky_feats = df[['ip', 'link', 'article_id', 'username', 'count_ip', 'count_links', 'count_usernames', 'article_id_flag']]

leaky_feats_train = leaky_feats.iloc[:train.shape[0]]

leaky_feats_test = leaky_feats.iloc[train.shape[0]:]

# filterout the entries without ips
train_ips = leaky_feats_train.ip[leaky_feats_train.count_ip != 0]
test_ips = leaky_feats_test.ip[leaky_feats_test.count_ip != 0]

train_ip_list = list(set([a for b in train_ips.tolist() for a in b]))
test_ip_list = list(set([a for b in test_ips.tolist() for a in b]))

common_ip_list = list(set(train_ip_list).intersection(test_ip_list))

plt.title("Common IP addresses")
venn.venn2(subsets=(len(train_ip_list), len(test_ip_list), len(common_ip_list)), set_labels=('# of unique IP in train', "# of unique IP in test"))

plt.savefig("Common IP addresses.png")


train_links = leaky_feats_train.link[leaky_feats_train.count_links != 0]

test_links = leaky_feats_test.link[leaky_feats_test.count_links != 0]

train_links_list = list(set([a for b in train_links.tolist() for a in b]))
test_links_list = list(set([a for b in test_links.tolist() for a in b]))

common_links_list = list(set(train_links_list).intersection(test_links_list))

plt.title("Common links")
plt.savefig("Common links.png")

venn.venn2(subsets=(len(train_links_list), len(test_links_list), len(common_links_list)), 
           set_labels=('# of unique links in train', "# of unique links in test"))

plt.title("Common links where entries without links removed")

plt.savefig("Common links where entries without links removed.png")

# filter out entries without users
train_users = leaky_feats_train.username[leaky_feats_train.count_usernames != 0]

test_users = leaky_feats_test.username[leaky_feats_test.count_usernames != 0]

train_users_list = list(set([a for b in train_users.tolist() for a in b]))
test_users_list = list(set([a for b in test_users.tolist() for a in b]))

common_users_list = list(set(train_users_list).intersection(test_users_list))
plt.title("Common usernames")
venn.venn2(subsets=(len(train_users_list), len(test_users_list), len(common_users_list)), 
           set_labels=("# of unique Usernames in train", "# of unique usernames in test"))


plt.savefig("Common usernames.png")

corpus = merge.comment_text
# corpus will look like
# 0         Explanation\nWhy the edits made under my usern...
# 1         D'aww! He matches this background colour I'm s...
# where the last row is 153,163

APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
}

def clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    # convert to lower case, so that Hi and hi are the same
    comment = comment.lower()
    # remove \n
    comment = re.sub("\\n", "", comment)
    # remove leaky elements like ip, user
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", '', comment)
    # removing usernames
    comment = re.sub("\[\[.*\]", "", comment)

    words = tokenizer.tokenize(comment)

    # (') apostrophe replacement ie you're --> you are
    words = [APPO[word] if word in APPO else word for word in words]
    words = [lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in eng_stopwords]

    clean_sent = " ".join(words)
    return(clean_sent)

clean_corpus = corpus.apply(lambda x: clean(x))

tfv = TfidfVectorizer(min_df=200, max_features=10000,
           strip_accents='unicode', analyzer='word', ngram_range=(1,1),
           use_idf=1,smooth_idf=1, sublinear_tf=1,
           stop_words = 'english')

tfv.fit(clean_corpus)

features = np.array(tfv.get_feature_names())

train_unigrams = tfv.transform(clean_corpus.iloc[:train.shape[0]])
test_unigrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])

def top_tfidf_feats(row, features, top_n=25):
    """
    Get top n tfidf values in row and return them with their corresponding feature names
    """
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    """
    Top tfidf features in specific document (matrix row)
    """
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids, min_tfidf=0.1, top_n=25):
    """
    Return the top n features that on average are most important amongst 
    documents in rows identified by indices in grp_ids
    """
    D = Xtr[grp_ids].toarray()
    D[D<min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, features, min_tfidf=0.1, top_n=20):
    """
    Return a list of dfs, where each df holds top_n features and their mean tfidf value
    calculated across documents with the same class label
    """
    dfs = []
    cols = train_tags.columns
    for col in cols:
        ids = train_tags.index[train_tags[col]==1]
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs


tfidf_top_n_per_lass = top_feats_by_class(train_unigrams, features)

tfv = TfidfVectorizer(min_df=150, max_features=30000,
            strip_accents='unicode', analyzer='word', ngram_range=(2,2),
            use_idf=1, smooth_idf=1, sublinear_tf=1,
            stop_words='english')

tfv.fit(clean_corpus)
features = np.array(tfv.get_feature_names())
train_bigrams = tfv.transform(clean_corpus.iloc[:train.shape[0]])
test_bigrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])

tfidf_top_n_per_lass = top_feats_by_class(train_bigrams, features)

tfv = TfidfVectorizer(min_df=100, max_features=30000,
            strip_accents='unicode', analyzer='char', ngram_range=(1,4),
            use_idf=1, smooth_idf=1, sublinear_tf=1,
            stop_words= 'english')

tfv.fit(clean_corpus)
features = np.array(tfv.get_feature_names())
train_charngrams = tfv.transform(clean_corpus.iloc[:train.shape[0]])
test_charngrams = tfv.transform(clean_corpus.iloc[train.shape[0]:])



