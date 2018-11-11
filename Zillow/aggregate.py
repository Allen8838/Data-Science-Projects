# from SRK notebook

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

#pd.options.mode.chained_assignment = raise
pd.options.display.max_columns = 999

prop_df = pd.read_csv('data/properties_2016.csv')

properties_2017 = pd.read_csv('data/properties_2017.csv')

train_df = pd.read_csv('data/train_2016_v2.csv', parse_dates=['transactiondate'])

plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('logerror', fontsize=12)
plt.savefig('Plot of Log error.png')

ulimit = np.percentile(train_df.logerror.values, 99)
llimit = np.percentile(train_df.logerror.values, 1)

# setting any values larger than the 99th percentile value to a fixed value
train_df['logerror'].ix[train_df['logerror']>ulimit] = ulimit

# setting any values lower than the 1 percentile value to a fixed value
train_df['logerror'].ix[train_df['logerror']<llimit] = llimit

plt.figure(figsize=(12,8))
sns.distplot(train_df.logerror.values, bins=50, kde=False)
plt.xlabel('logerror', fontsize=12)

plt.savefig('Histogram of log error.png')

train_df['transaction_month'] = train_df['transactiondate'].dt.month

cnt_srs = train_df['transaction_month'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
plt.xlabel('Month of transaction', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.savefig('Number of transactions in 2016 by month.png')


# this creates a 2 column dataframe where the first column are the 
# categories of the dataset and the second column is a count of
# how many rows in that column have null values 
missing_df = prop_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

# filter for the rows with actual null values
missing_df = missing_df.ix[missing_df['missing_count']>0]

# sorts the missing count value in ascending order
missing_df = missing_df.sort_values(by='missing_count')

# creates a row matrix of the index values 
# [0, 1, 2, 3, .... 55, 56]
ind = np.arange(missing_df.shape[0])

width = 0.9
fig, ax = plt.subplots(figsize=(12, 18))
rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")

plt.savefig('Number of missing values in each column in training 2016 data.png')

train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')

pd.options.display.max_rows = 65

# returns two columns. one column of the datatype in the train_df
# the second column of the datatype for each row in the first column
dtype_df = train_df.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

# reset index at the end will include the leftmost side default
# row count as an index
dtype_df.groupby("Column Type").aggregate('count').reset_index()


missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['missing_ratio'] = missing_df['missing_count']/train_df.shape[0]
missing_df.ix[missing_df['missing_ratio']>0.999]

mean_values = train_df.mean(axis=0)

# quick and dirty way of filling each missing value with the respective 
# mean values
train_df_new = train_df.fillna(mean_values)

# collects the column headers that does not equal to the logerror and has a datatype of float64 in train_df_new
x_cols = [col for col in train_df_new.columns if col not in ['logerror'] if train_df_new[col].dtype=='float64']

labels = []
values = []

for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(train_df_new[col].values, train_df_new.logerror.values)[0, 1]) # the [0, 1] at the end collects the upper left quadrant of the 
                                                                                             # resulting list of arrays where each array is a 2 x 2 matrix
corr_df = pd.DataFrame({'col_labels': labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')

ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,40))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel('Correlation coefficient')
ax.set_title('Correlation coefficient of the variables')

plt.savefig('Correlation coefficient of the variables for 2016 traning set.png')

corr_zero_cols = ['assessmentyear', 'storytypeid', 'pooltypeid2', 'pooltypeid7', 'pooltypeid10', 'poolcnt', 'decktypeid', 'buildingclasstypeid']

# from the resulting Correlation coefficient of the variables for 2016 traning set.png we suspect that
# corr_zero_cols are composed of unique values. let's print these out to see if that's really the case
for col in corr_zero_cols:
    print(col, len(train_df_new[col].unique()))

# selected dataframe with high correlations 
corr_df_sel = corr_df.ix[(corr_df['corr_values']>0.02)| (corr_df['corr_values'] < -0.01)]


