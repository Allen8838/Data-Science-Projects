import operator
from plotly.offline import plot
import plotly.graph_objs as go

census_2013 = {'Mississippi': 2991207, 'Iowa': 3090416, 'Oklahoma': 3850568, 'Delaware': 925749, 'Minnesota': 5420380, 'Alaska': 735132, 'Illinois': 12882135, 'Arkansas': 2959373, 'New Mexico': 2085287,
'Indiana': 6570902, 'Maryland': 5928814, 'Louisiana': 4625470, 'Texas': 26448193, 'Wyoming': 582658, 'Arizona': 6626624, 'Wisconsin': 5742713, 'Michigan': 9895622, 'Kansas': 2893957, 'Utah': 2900872,
'Virginia': 8260405, 'Oregon': 3930065, 'Connecticut': 3596080, 'New York': 19651127, 'California': 38332521, 'Massachusetts': 6692824, 'West Virginia': 1854304, 'South Carolina': 4774839, 'New Hampshire': 1323459,
'Vermont': 626630, 'Georgia': 9992167, 'North Dakota': 723393, 'Pennsylvania': 12773801, 'Florida': 19552860, 'Hawaii': 1404054, 'Kentucky': 4395295, 'Rhode Island': 1051511, 'Nebraska': 1868516, 'Missouri': 6044171,
'Ohio': 11570808, 'Alabama': 4833722, 'South Dakota': 844877, 'Colorado': 5268367, 'Idaho': 1612136, 'New Jersey': 8899339, 'Washington': 6971406, 'North Carolina': 9848060, 'Tennessee': 6495978, 'Montana': 1015165,
'District of Columbia': 646449, 'Nevada': 2790136, 'Maine': 1328302}



def create_plot_of_population_per_100k(df, column):
    donors_from_states = dict(df[column].value_counts())
    
    don_pop = {}

    for state, don in donors_from_states.items():
        if state not in census_2013:
            continue
        don_pop[state] = float(don)*100000/census_2013[state]
    
    # sort the dictionary so that the highest donors per 100K goes first
    don_pop = sorted(don_pop.items(), key=operator.itemgetter(1), reverse=True)

    xx = [x[0] for x in (don_pop)][1:]
    yy = [x[1] for x in (don_pop)][1:]


    trace2 = go.Bar(
        x = xx,
        y = yy,
        name = 'Donors to Population',
        marker = dict(color='green'),
        opacity = 0.3
    )

    data = [trace2]
    layout = go.Layout(
        barmode = 'group',
        legend = dict(dict(x=-0.1, y=1.2)),
        margin = dict(b=120),
        title = 'States with highest Donors to Population Ratio'
    )

    # plots figure to browser
    plot(go.Figure(data=data, layout=layout))
    
    return None

