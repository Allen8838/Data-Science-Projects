from plot import plot_to_us_map,\
                 plot_teachers_posting_first_project,\
                 plot_distribution_of_project_type_and_status,\
                 plot_funded_amount_by_states,\
                 plot_project_title_to_world_cloud,\
                 plot_time_taken_to_fully_fund_projects,\
                 plot_funded_projects_by_states
from census import create_plot_of_population_per_100k
from text_cleaning import clean_txt, generate_ngrams
from calculate import calculate_project_costs_by_metro_type, find_distribution_of_project_cost_by_metro_type, calculate_donations_from_home_state
from topic_modelling import lda_modelling

import itertools
from collections import Counter

from plotly.offline import init_notebook_mode, iplot
from wordcloud import wordcloud
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly import tools
from datetime import date
import pandas as pd
import numpy as np
import seaborn as sns
import random
import warnings
import operator
warnings.filterwarnings("ignore")
init_notebook_mode(connected=False)

donors_df = pd.read_csv("Donors.csv")
donations_df = pd.read_csv("Donations.csv")
teachers_df = pd.read_csv("Teachers.csv")

projects_df = pd.read_csv('Projects.csv')
projects_df['Posted Date']  = pd.to_datetime(projects_df['Project Posted Date'])
projects_df['Posted Year'] = projects_df['Posted Date'].dt.year
projects_df['Posted Month'] = projects_df['Posted Date'].dt.month

schools_df = pd.read_csv('Schools.csv')

#teachers_df['Posted Date'] = pd.datetime(teachers_df['Teacher First Project Posted Date'])

plot_to_us_map(donors_df, 'Donor State')

create_plot_of_population_per_100k(donors_df, 'Donor State')

plot_teachers_posting_first_project(teachers_df, 'Teacher First Project Posted Date')


#plot_distribution_of_project_type_and_status(projects_df, 'Project Type', 'Project Current Status')

projsch_df = projects_df.merge(schools_df, on="School ID")

plot_funded_amount_by_states(projsch_df)


# Bag of words analysis
projects_df = projects_df.merge(schools_df, on='School ID', how='inner')
april18_df = projects_df[(projects_df['Posted Year']==2018) & (projects_df['Posted Month']==4)]

april18_df['clean_essay'] = april18_df['Project Essay'].apply(clean_txt)
april18_df['clean_need'] = april18_df['Project Need Statement'].apply(clean_txt)

april18_df['unigrams'] = april18_df['clean_essay'].apply(lambda x : generate_ngrams(x.split(), 1))
april18_df['bigrams'] = april18_df['clean_essay'].apply(lambda x : generate_ngrams(x.split(), 2))
april18_df['trigrams'] = april18_df['clean_essay'].apply(lambda x : generate_ngrams(x.split(), 3))

all_unigrams = []
for each in april18_df['unigrams']:
    all_unigrams.extend(each)

t = Counter(all_unigrams).most_common(25)
x = [a[0] for a in t]
y = [a[1] for a in t]

all_bigrams = []
for each in april18_df['bigrams']:
    all_bigrams.extend(each)

t1 = Counter(all_bigrams).most_common(25)
x1 = [a[0] for a in t1]
y1 = [a[1] for a in t1]

all_trigrams = []
for each in april18_df['trigrams']:
    all_trigrams.extend(each)

t2 = Counter(all_trigrams).most_common(25)
x2 = [a[0] for a in t2]
y2 = [a[1] for a in t2]

sns.set_style('dark')

fig, axes = plt.subplots(nrows=1, ncols=5, squeeze=True, figsize = (12, 10))

bar = sns.barplot(y=x, x=y, ax=axes[0], palette='GnBu_d', edgecolor='white')
bar.set(xlabel='', xticks=[])
axes[0].set_title("Top Keywords used by Teachers")

fig.delaxes(axes[1])

bar1 = sns.barplot(y=x1, x=y1, ax=axes[2], palette='GnBu_d', edgecolor='white')
bar1.set(xlabel='', xticks=[])
axes[2].set_title("Top Bigrams used by Teachers")

fig.delaxes(axes[3])

bar1 = sns.barplot(y=x2, x=y2, ax=axes[4], palette='GnBu_d', edgecolor='white')
bar1.set(xlabel='', xticks=[])
axes[4].set_title("Top Trigrams used by Teachers")

fig.savefig('Ngrams.png')

#plot_project_title_to_world_cloud(april18_df)

plot_time_taken_to_fully_fund_projects(projects_df)

plot_funded_projects_by_states(projects_df, schools_df)

calculate_project_costs_by_metro_type(projects_df, schools_df)

find_distribution_of_project_cost_by_metro_type(projects_df, schools_df)

calculate_donations_from_home_state(donors_df, donations_df, projects_df, schools_df)

lda_modelling(projects_df)