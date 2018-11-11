# from gloria Hristova kernel

# Data pre-processing
import pandas as pd
import json
from collections import Counter
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

# data visualizations
import random
import plotly
from plotly import tools
from plotly.offline import download_plotlyjs, plot
import plotly.offline as offline
import plotly.graph_objs as go

# data modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import model_selection
import warnings
warnings.filterwarnings('ignore')

train_data = pd.read_json('data/train.json')
test_data = pd.read_json('data/test.json')

print(train_data.info())

print('Number of unique dishes: {}'.format(len(train_data.cuisine.unique())))

def random_colours(number_of_colors):
    '''
    number_of_colors - number of colors that are generated
    Output: color in the following format ['#E86DA4']
    '''
    colors = []
    for i in range(number_of_colors):
        colors.append('#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
    return colors

trace = go.Table(
                header=dict(values=['Cuisine', 'Number of recipes'],
                fill = dict(color=['#EABEB0']),
                align = ['left']*5),
                cells = dict(values=[train_data.cuisine.value_counts().index, train_data.cuisine.value_counts()],
                align = ['left']*5))

layout = go.Layout(title='Number of recipes in each cuisine category',
                   titlefont = dict(size = 20),
                   width = 500, height = 650,
                   paper_bgcolor = 'rgba(0, 0, 0, 0)',
                   plot_bgcolor = 'rgba(0, 0, 0, 0)',
                   autosize = False,
                   margin = dict(l=30, r=30, b=1, t=50, pad=1),)

data = [trace]
fig = dict(data=data, layout=layout)
plot(fig)

labelpercents = []
for i in train_data.cuisine.value_counts():
    percent = (i/sum(train_data.cuisine.value_counts()))*100
    # rounds to the nearest 2 decimal places
    percent = '%.2f' % percent
    percent = str(percent + '%')
    labelpercents.append(percent)

trace = go.Bar(
            x = train_data.cuisine.value_counts().values[::-1],
            y = [i for i in train_data.cuisine.value_counts().index][::-1],
            text = labelpercents[::-1], # [::-1] start at the end, count down to the beginning, stepping backwards one step at a time
            orientation = 'h', marker = dict(color = random_colours(20)))

layout = go.Layout(
            title = 'Number of recipes in each cuisine category',
            titlefont = dict(size=25),
            width = 1000, height = 450, 
            plot_bgcolor = 'rgba(0, 0, 0, 0)',
            paper_bgcolor = 'rgba(255, 219, 227, 0.88)',
            margin = dict(l=75, r=110, b=50, t=60),)

data = [trace]
fig = dict(data=data, layout=layout)
plot(fig)

