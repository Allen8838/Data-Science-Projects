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

trace = go.Histogram(
           x = train_data['ingredients'].str.len(),
           xbins = dict(start=0, end=90, size=1),
           marker = dict(color= '#7CFDF0'),
           opacity = 0.75)

data = [trace]

layout = go.Layout(
           title = 'Distribution of Recipe Length',
           xaxis = dict(title='Number of ingredients'),
           yaxis = dict(title= 'Count of recipes'),
           bargap = 0.1,
           bargroupgap = 0.2)

fig = go.Figure(data=data, layout=layout)
plot(fig)

boxplotcolors = random_colours(21)
labels = [i for i in train_data.cuisine.value_counts().index][::-1]
data = []

for i in range(20):
    trace = go.Box(
                y = train_data[train_data['cuisine'] == labels[i]]['ingredients'].str.len(), name= labels[i],
                marker = dict(color=boxplotcolors[i]))
    data.append(trace)

layout = go.Layout(
                title = 'Recipe Length Distribution by cuisine'
)

fig = go.Figure(data=data, layout=layout)

plot(fig)

# stores all ingredients in all recipes (with duplicates)
allingredients = []
for item in train_data['ingredients']:
    for ingr in item:
        allingredients.append(ingr)

countingr = Counter()
for ingr in allingredients:
    countingr[ingr] += 1

mostcommon = countingr.most_common(20)
mostcommoningr = [i[0] for i in mostcommon]
mostcommoningr_count = [i[1] for i in mostcommon]

trace = go.Bar(
            x = mostcommoningr_count[::-1],
            y = mostcommoningr[::-1],
            orientation = 'h', marker = dict(color = random_colours(20),
            ))

layout = go.Layout(
            xaxis = dict(title='Number of occurences in all recipes (training sample)',),
            yaxis = dict(title='Ingredient'),
            title = '20 Most Common Ingredients', titlefont = dict(size=20),
            margin = dict(l=150, r=10, b=60, pad=5),
            width = 800, height = 500,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
plot(fig)

def findnumingr(cuisine):
    '''
    Input: 
        cuisine - cuisine category (e.g. greek, chinese, etc.)
    Output:
        The number of unique ingredients used in all recipes part of the given cuisine
    '''
    listofingr = []
    for item in train_data[train_data['cuisine'] == cuisine]['ingredients']:
        for ingr in item:
            listofingr.append(ingr)
    result = (cuisine, len(list(set(listofingr))))
    return result


cuisineallingr = []
for i in labels:
    cuisineallingr.append(findnumingr(i))

trace = go.Bar(
            x = [i[1] for i in cuisineallingr],
            y = [i[0] for i in cuisineallingr],
            orientation = 'h', marker = dict(color = random_colours(20),))

layout = go.Layout(
            xaxis = dict(title = 'Count of different ingredients', ),
            yaxis = dict(title = 'Cuisine', ),
            title = 'Number of all the different ingredients used in a given cuisine', titlefont = dict(size=20),
            margin = dict(l=100, r=10, b=60, t=60),
            width = 800, height = 500,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
plot(fig)

# which ingredient occur in only one cuisine

allingredients = list(set(allingredients)) # get all unique ingredients

def cuisine_unique(cuisine, numingr, allingredients):
    '''
    Input:
        cuisine - cuisine category (e.g. chinese)
        numingr - how many specific ingredients do you want to see in the final result
        allingredients - list containing all unique ingredients in the whole sample
    Output:
        dataframe giving information about the name of the specific ingredient and how many times it
        occurs in the chosen cuisine (in descending order based on their counts)
    '''
    allother = []
    for item in train_data[train_data.cuisine != cuisine]['ingredients']:
        for ingr in item:
            allother.append(ingr)
    allother = list(set(allother))

    specificonly = [x for x in allingredients if x not in allother]

    mycounter = Counter()

    for item in train_data[train_data.cuisine == cuisine]['ingredients']:
        for ingr in item:
            mycounter[ingr] += 1
    keep = list(specificonly)

    for word in list(mycounter):
        if word not in keep:
            del mycounter[word]
    
    cuisinespec = pd.DataFrame(mycounter.most_common(numingr), columns = ['ingredient', 'count'])

    return cuisinespec

cuisinespec = cuisine_unique('chinese', 10, allingredients)
# print('The top 10 unique ingredients in Chinese cuisine are: ', cuisinespec)

# Visualization of specific ingredients in the first 10 cuisines
labels = [i for i in train_data.cuisine.value_counts().index][0:10]
totalPlot = 10
y = [[item]*2 for item in range(1, 10)]
y = list(chain.from_iterable(y))
z = [1,2]*int((totalPlot/2))

fig = tools.make_subplots(rows=5, cols=2, subplot_titles=labels, specs= [[{}, {}], [{}, {}], [{}, {}], [{}, {}], [{}, {}]], horizontal_spacing=0.20)

traces = []

for i, e in enumerate(labels):
    cuisinespec = cuisine_unique(e, 5, allingredients)
    trace = go.Bar(
            x = cuisinespec['count'].values[::-1],
            y = cuisinespec['ingredient'].values[::-1],
            orientation = 'h', marker = dict(color=random_colours(5),))
    
    traces.append(trace)

for t, y, z in zip(traces, y, z):
    fig.append_trace(t, y, z)
    fig['layout'].update(height=800, width=840, margin=dict(l=265, r=5, b=40, t=90, pad=5), showlegend=False, title='Ingredients used only in one cuisine')

plot(fig)

# TF-IDF
features = []
for item in train_data['ingredients']:
    features.append(item)

ingredients = []
for item in train_data['ingredients']:
    for ingr in item:
        ingredients.append(ingr)

# Fit the TfidfVectorizer to data
tfidf = TfidfVectorizer(vocabulary=list(set([str(i).lower() for i in ingredients])), max_df=0.99, norm='12', ngram_range=(1, 4))

X_tr = tfidf.fit_transform([str(i) for i in features])
feature_names = tfidf.get_feature_names()

def top_feats_by_class(trainsample, target, featurenames, min_tfidf=0.1, top_n=10):
    '''
    Input:
        trainsample - the tf-idf transformed training sample
        target - the target variable
        featurenames - 
        min_tfidf - features having tf-idf value below the min_tfidf will be excluded
        top_n - how many important features to show
    
    Output:
        Returns a list of dataframe objects, where each dataframe holds top_n features and their mean tfidf value
        calculated across documents (recipes) with the same class label (cuisine)
    '''
    dfs = []
    labels = np.unique(target)

    for label in labels:

        ids = np.where(target==label)
        D = trainsample[ids].toarray()
        D[D< min_tfidf] = 0
        tfidf_means = np.nanmean(D, axis=0)

        topn_ids = np.argsort(tfidf_means)[::-1][top_n]
        top_feats = [(feature_names[i], tfidf_means[i]) for i in topn_ids]
        df = pd.DataFrame(top_feats)
        df.columns = ['feature', 'tfidf']

        df['cuisine'] = label
        dfs.append(df)

    return dfs

target = train_data['cuisine']

result_tfidf = top_feats_by_class(X_tr, target, feature_names, min_tfidf=0.1, top_n=5)


# Feature importance according to Tf-Idf measure
labels = []

for i, e in enumerate(result_tfidf):
    labels.append(result_tfidf[i].cuisine[0])

# set the plot
totalPlot = 10
y = [[item]*2 for item in range(1,10)]
y = list(chain.from_iterable(y))
z = [1,2]*int((totalPlot/2))

fig = tools.make_subplots(rows=5, cols=2, subplot_titles=labels[0:10], spec=[[{}, {}], [{}, {}], [{}, {}], [{}, {}], [{}, {}]], horizontal_spacing=0.20)

traces = []

for index, element in enumerate(result_tfidf[0:10]):
    trace = go.Bar(
            x = result_tfidf[index].tfidf[::-1],
            y = result_tfidf[index].feature[::-1],
            orientation = 'h', marker = dict(color = random_colours(5),))

    traces.append(trace)

for t, y, z in zip(traces, y, z):
    fig.append(t, y, z)

    fig['layout'].update(height=800, width=840,
    margin=dict(l=110, r=5, b=40, t=90, pad=5), showlegend=False, title='Feature Importance based on Tf-Idf measure')

plot(fig)

