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
from sklearn.svc import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import model_selection
import warnings
warnings.filterwarnings('ignore')

