import folium
from folium import plugins
from io import StringIO
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot

statesll=StringIO("""State,Latitude,Longitude
Alabama,32.806671,-86.791130
Alaska,61.370716,-152.404419
Arizona,33.729759,-111.431221
Arkansas,34.969704,-92.373123
California,36.116203,-119.681564
Colorado,39.059811,-105.311104
Connecticut,41.597782,-72.755371
Delaware,39.318523,-75.507141
District of Columbia,38.897438,-77.026817
Florida,27.766279,-81.686783
Georgia,33.040619,-83.643074
Hawaii,21.094318,-157.498337
Idaho,44.240459,-114.478828
Illinois,40.349457,-88.986137
Indiana,39.849426,-86.258278
Iowa,42.011539,-93.210526
Kansas,38.526600,-96.726486
Kentucky,37.668140,-84.670067
Louisiana,31.169546,-91.867805
Maine,44.693947,-69.381927
Maryland,39.063946,-76.802101
Massachusetts,42.230171,-71.530106
Michigan,43.326618,-84.536095
Minnesota,45.694454,-93.900192
Mississippi,32.741646,-89.678696
Missouri,38.456085,-92.288368
Montana,46.921925,-110.454353
Nebraska,41.125370,-98.268082
Nevada,38.313515,-117.055374
New Hampshire,43.452492,-71.563896
New Jersey,40.298904,-74.521011
New Mexico,34.840515,-106.248482
New York,42.165726,-74.948051
North Carolina,35.630066,-79.806419
North Dakota,47.528912,-99.784012
Ohio,40.388783,-82.764915
Oklahoma,35.565342,-96.928917
Oregon,44.572021,-122.070938
Pennsylvania,40.590752,-77.209755
Rhode Island,41.680893,-71.511780
South Carolina,33.856892,-80.945007
South Dakota,44.299782,-99.438828
Tennessee,35.747845,-86.692345
Texas,31.054487,-97.563461
Utah,40.150032,-111.862434
Vermont,44.045876,-72.710686
Virginia,37.769337,-78.169968
Washington,47.400902,-121.490494
West Virginia,38.491226,-80.954453
Wisconsin,44.268543,-89.616508
Wyoming,42.755966,-107.302490""")

def plot_to_us_map(df, column):
    tempdf = df[column].value_counts()
    t1 = pd.DataFrame()
    t1[column] = tempdf.index
    t1[column+' Count'] = tempdf.values

    sdf = pd.read_csv(statesll).rename(columns={'State':'Donor State'})
    sdf = sdf.merge(t1, on='Donor State', how='inner')

    map4 = folium.Map(location=[39.50, -98.35], tiles='CartoDB dark_matter', zoom_start=3.5)

    for j, row in sdf.iterrows():
        row = list(row)
        folium.CircleMarker([float(row[1]), float(row[2])], popup='<b>State:</b>' + row[0].title()+'<br> <b>Donors:</b> '+str(int(row[3])), radius=float(row[3])*0.0001, color='#be0eef', fill=True).add_to(map4)

    map4.save('Donor State.html')
    return None


def bar_ver_noagg(x, y, title, color, w=None, h=None, lm=0, rt=False):
    trace = go.Bar(y=y, x=x, marker=dict(color=color))

    if rt:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)

    plot(fig)

def plot_teachers_posting_first_project(df, column):
    t = df[column].value_counts()
    x = t.index
    y = t.values
    bar_ver_noagg(x, y, 'Date & Teacher First Projects', 'orange')

def plot_distribution_of_project_type_and_status(df, project_type, project_status):
    temp = df[project_type].value_counts()
    values1 = temp.values
    index1 = temp.index

    temp = df[project_status].value_counts()
    values2 = temp.values
    index2 = temp.index

    domain1 = {'x': [0.2, 0.50], 'y':[0.0, 0.33]}
    domain2 = {'x': [0.8, 0.50], 'y':[0.0, 0.33]}

    fig = {
        "data": [
            {
                "values": values1,
                "labels": index1,
                "domain": {"x": [0, .48]},
                "marker": dict(colors=["#f77b9c", "#ab97db", '#b0b1b2']),
                "name": "Project Type",
                "hoverinfo": "label+percent+name",
                "hole": 0.7,

                "type": "pie"
            },
            {
                "values":values2,
                "labels": index2,
                "marker": dict(colors=["#efbc56", "#81a7e8", "#e295d0"]),
                "domain": {'x': [0.52, 1]},
                #"text": "CO2",
                "textposition":"inside",
                "name": "Project Status",
                "hole": 0.7,
                "type": "pie"
            }],
            "layout":{
                "annotations": [
                    {
                        "font":{
                            "size":20
                        },
                        "showarrow":False,
                        "text": "Type",
                        "x": 0.21,
                        "y": 0.5
                    },
                    {
                        "font": {
                            "size": 20
                        },
                        "showarrow": False,
                        "text": "Status",
                        "x": 0.8,
                        "y": 0.5
                    }
                ]
            }
    }
    plot(fig, filename='donut')


