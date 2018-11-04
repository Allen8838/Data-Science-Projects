from plot import plot_to_us_map
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
init_notebook_mode(connected=True)

donors_df = pd.read_csv("Donors.csv")
donations_df = pd.read_csv("Donations.csv")
teachers_df = pd.read_csv("Teachers.csv")

map4 = plot_to_us_map(donors_df, 'Donor State')

map4