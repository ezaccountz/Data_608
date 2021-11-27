# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 18:35:08 2021

@author: Euclid
"""

import dash
#import dash_core_components as dcc
#import dash_html_components as html
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px


import datashader as ds, datashader.transfer_functions as tf
#import numpy as np
#from datashader import spatial
#from datashader.utils import lnglat_to_meters as webm
from functools import partial
from datashader.utils import export_image
from datashader.colors import colormap_select#, Greys9
from colorcet import fire
import pandas as pd
import math
import re


app_token = 'HcYUIkQM18kU32TiVXYsAKVjy'
address = 'https://data.cityofnewyork.us/resource/h9gi-nx95.json'
select_columns = ['date_extract_y(crash_date)%20as%20year','date_extract_m(crash_date)%20as%20month',
                  'crash_time','latitude','longitude',
                  'contributing_factor_vehicle_1%20as%20factor','vehicle_type_code1%20as%20vehicle_type']

limit = 999999
dfs = dict()

factor_cov = pd.read_csv('https://raw.githubusercontent.com/ezaccountz/Data_608/main/Final_Project/factors.csv')  
factor_cov = dict(zip(factor_cov['original factors'], factor_cov['revised factors']))

vehicle_type_cov = pd.read_csv('https://raw.githubusercontent.com/ezaccountz/Data_608/main/Final_Project/vehicle_types.csv', keep_default_na=False)
vehicle_type_cov = dict(zip(vehicle_type_cov['original vehicle types'], vehicle_type_cov['revised vehicle types']))

url = address + '?$select=distinct(date_extract_y(crash_date))%20as%20year'
url = url + "&$$app_token="+app_token
years = pd.read_json(url)
years.sort_values('year', ascending=False, inplace = True)
years = years['year'].to_list()

months = list(range(1,13))
months.insert(0,'All')
hours = list(range(1,25))
hours.insert(0,'All')

url = address + '?$select=distinct(contributing_factor_vehicle_1)%20as%20factors'
url = url + "&$limit="+str(limit)
url = url + "&$$app_token="+app_token
factors = pd.read_json(url)
factors.replace(factor_cov, inplace = True)
factors = factors['factors'].dropna().to_list()
factors = list(set(factors))
#factors = np.insert(factors,0,'All')
factor_list = dict()
for factor in factors:
    factor_list[factor] = []
for key, item in factor_cov.items():
    factor_list[item].append(key)

url = address + '?$select=distinct(vehicle_type_code1)%20as%20vehicle_types'
url = url + "&$limit="+str(limit)
url = url + "&$$app_token="+app_token
vehicle_types = pd.read_json(url)
vehicle_types['vehicle_types'] = vehicle_types['vehicle_types'].str.upper()
vehicle_types['vehicle_types'].fillna('Unspecified', inplace = True)
vehicle_types['vehicle_types']  = vehicle_types['vehicle_types'].map(lambda x: re.sub(r'\W+', ' ', x))
vehicle_types['vehicle_types'] = vehicle_types['vehicle_types'].str.strip()
vehicle_types.replace(vehicle_type_cov, inplace = True)
vehicle_types = vehicle_types['vehicle_types'].to_list()
vehicle_types = list(set(vehicle_types))
#vehicle_types = np.insert(vehicle_types,0,'All')


#NewYorkCity   = (( -74.25909,  -73.700181), (40.477399, 40.916178))
NewYorkCity   = (( -74.25909,  -73.700181), (40.487399, 40.926178))

#-----------------------------------------------------------------------------



#background color
background = "black"
#export_image: partial helper function
export = partial(export_image, background = background, export_path="export")
#colormap_select: partial helper function
cm = partial(colormap_select, reverse=(background!="black"))
#plot width and height
plot_width  = int(800)
#plot_height = int(plot_width*7.0/12)
plot_height = int(plot_width*0.421)


def getDF(selected_year):
    if not (selected_year in dfs):
        url = address
        url = url + "?$select="+",".join(select_columns)
        url = url + "&$where=year=" + str(selected_year)
        url = url + "&$limit="+str(limit)
        url = url + "&$$app_token="+app_token
        dfs[selected_year] = pd.read_json(url)
        #TO DO: fix NAs
        
        dfs[selected_year]['hour'] = dfs[selected_year]['crash_time'].dt.hour
        dfs[selected_year].drop(columns = ['crash_time'],inplace = True)
        
        dfs[selected_year]['factor'].replace(factor_cov, inplace = True)
        dfs[selected_year]['factor'].fillna('Unspecified', inplace = True)
              
        dfs[selected_year]['vehicle_type'] = dfs[selected_year]['vehicle_type'].str.upper()
        dfs[selected_year]['vehicle_type'].fillna('Unspecified', inplace = True)
        #dfs[selected_year]['vehicle_type']  = dfs[selected_year]['vehicle_type'].map(lambda x: re.sub(r'\W+', ' ', x).strip())
        
        dfs[selected_year]['vehicle_type']  = dfs[selected_year]['vehicle_type'].map(lambda x: re.sub(r'\W+', ' ', x))
        dfs[selected_year]['vehicle_type'] = dfs[selected_year]['vehicle_type'].str.strip()
        #dfs[selected_year]['vehicle_type'] = dfs[selected_year]['vehicle_type'].apply(lambda x: str(x).encode('ascii','ignore').decode())
        dfs[selected_year]['vehicle_type'].replace(vehicle_type_cov, inplace = True)
        
        
        dfs[selected_year].dropna(inplace = True)
      
    return dfs[selected_year]

def create_image(df, longitude_range, latitude_range, w=plot_width, h=plot_height, zoom = 10):
    #create cvs
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=longitude_range, y_range=latitude_range)
    #Aggregation by category
    agg = cvs.points(df, 'longitude', 'latitude')
    #create image
    img = tf.shade(agg, cmap=fire, how='eq_hist')     
    img = tf.spread(img, px=math.floor(zoom/10.5))
    
    img = img[::-1].to_pil()
    
    coords_lat, coords_lon = agg.coords['latitude'].values, agg.coords['longitude'].values
    coordinates = [[coords_lon[0], coords_lat[0]],
               [coords_lon[-1], coords_lat[0]],
               [coords_lon[-1], coords_lat[-1]],
               [coords_lon[0], coords_lat[-1]]]
    
    xcenter = [(coords_lon[0]+coords_lon[-1])/2]
    ycenter = [(coords_lat[0]+coords_lat[-1])/2]
    
    geo_fig = px.scatter_mapbox(lat=ycenter, 
                        lon=xcenter, 
                        size = [0],
                        width=plot_width, height=plot_height, zoom = zoom)
                        #width=plot_width, height=plot_height,zoom=10)
    geo_fig.update_layout(mapbox_style="carto-darkmatter",
                    mapbox_layers = [
                        {
                            "sourcetype": "image",
                            "source": img,
                            "coordinates": coordinates
                        }
                    ],
                    hovermode=False,
                    margin=dict(t=0,b=0,l=10,r=10)
    )   
    return geo_fig


getDF(years[0])


CONTENT_STYLE_50 = {
    "display": "inline-block",
    "width": "48vw"
}

CONTENT_STYLE_33 = {
    "display": "inline-block",
    "width": "32vw"
}

CONTENT_STYLE_66 = {
    "display": "inline-block",
    "width": "65vw"
}

CONTENT_STYLE_25 = {
    "display": "inline-block",
    "width": "24vw"
}

CONTENT_STYLE_75 = {
    "display": "inline-block",
    "width": "73vw"
}

CONTENT_STYLE_10 = {
    "display": "inline-block",
    "width": "9vw"
}

CONTENT_STYLE_90 = {
    "display": "inline-block",
    "width": "88vw"
}


#--------------------------------------------------------------

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
         html.Label(children = "Year: ",style = {'font-weight': 'bold',"text-align": "left","display": "inline-block","width": "4vw"}),   
         html.Div([
             dcc.Dropdown(
                        id="years_dropdown",
                        options=[{"label": x, "value": x} for x in years],
                        value=years[0],
                        clearable=False,
             ),             
         ],style = {"display": "inline-block", "width": "4vw"}),

    ],style=CONTENT_STYLE_10),
    html.Div([
        html.Label(children = "Factor: ",style = {'font-weight': 'bold',"text-align": "left","display": "inline-block","width": "8vw"}),
        html.Div([
             dcc.Dropdown(
                        id="factors_dropdown",
                        options=[{"label": x, "value": x} for x in factors],
                        multi=True,
                        placeholder="All factors",
                        #value=factors[0],
                        #clearable=False,
            ),             
        ],style = {"display": "inline-block", "width": "79vw"}),
        #html.Br,
        html.Label(children = "Vehicle Type: ",style = {'font-weight': 'bold',"text-align": "left","display": "inline-block","width": "8vw"}),
        html.Div([
             dcc.Dropdown(
                        id="vehicle_types_dropdown",
                        options=[{"label": x, "value": x} for x in vehicle_types],
                        multi=True,
                        placeholder="All vehicle types",
                        #value=vehicle_types[0],
                        #clearable=False,
            ),         
        ],style = {"display": "inline-block", "width": "79vw"}),
    ],style=CONTENT_STYLE_90),
    
     
    
    html.Div([
         dcc.Graph(id="months_plot"),
    ],style=CONTENT_STYLE_33),
    html.Div([
         dcc.Graph(id="hours_plot"),
    ],style=CONTENT_STYLE_66),  
    html.H3(id="total_all",style={'font-weight': 'bold',"text-align": "center","display": "inline-block","width": "48vw"}),    
    html.H3(id="total_filtered",style = {'font-weight': 'bold',"text-align": "center","display": "inline-block","width": "48vw"}),          
    html.Div([
         dcc.Graph(id="map_plot"),
    ],style=CONTENT_STYLE_50),
    html.Div([
         dcc.Graph(id="map_plot2"),
    ],style=CONTENT_STYLE_50),
    html.Div([
         dcc.Graph(id="factors_plot"),
    ],style=CONTENT_STYLE_25),
    html.Div([
         dcc.Graph(id="vehicle_types_plot"),
    ],style=CONTENT_STYLE_25),
    html.Div([
         dcc.Graph(id="factors_plot2"),
    ],style=CONTENT_STYLE_25),
    html.Div([
         dcc.Graph(id="vehicle_types_plot2"),
    ],style=CONTENT_STYLE_25),
    
 
])

 
@app.callback(
    Output('map_plot', 'figure'),
    Output('map_plot2', 'figure'),
    Output('months_plot', 'figure'),
    Output('hours_plot', 'figure'),
    Output('factors_plot', 'figure'),
    Output('vehicle_types_plot', 'figure'),
    Output('factors_plot2', 'figure'),
    Output('vehicle_types_plot2', 'figure'),
    Output("total_all", "children"), 
    Output("total_filtered", "children"), 
    [Input('map_plot', 'relayoutData'),
     Input('map_plot2', 'relayoutData'),
     Input('years_dropdown', 'value'),
     Input('factors_dropdown', 'value'),
     Input('vehicle_types_dropdown', 'value'),
    ],
)
def update_scatter_chart(m1_relayoutData, m2_relayoutData, year, factors, vehicle_types):
    
    dashcc = dash.callback_context
    dashcc = dashcc.triggered[0]['prop_id'].split('.')[0]
    
    collisions = getDF(year)
    
    x0 = NewYorkCity[0][0]
    x1 = NewYorkCity[0][1]
    y0 = NewYorkCity[1][0]
    y1 = NewYorkCity[1][1]   
    zoom = 8.7
    
    if dashcc == 'map_plot2':
        if m2_relayoutData is not None and 'mapbox._derived' in m2_relayoutData: 
            x0 = m2_relayoutData['mapbox._derived']['coordinates'][0][0]
            x1 = m2_relayoutData['mapbox._derived']['coordinates'][1][0]
            y0 = m2_relayoutData['mapbox._derived']['coordinates'][2][1]
            y1 = m2_relayoutData['mapbox._derived']['coordinates'][0][1]
            zoom = m2_relayoutData['mapbox.zoom']
    else:
        if m1_relayoutData is not None and 'mapbox._derived' in m1_relayoutData: 
            x0 = m1_relayoutData['mapbox._derived']['coordinates'][0][0]
            x1 = m1_relayoutData['mapbox._derived']['coordinates'][1][0]
            y0 = m1_relayoutData['mapbox._derived']['coordinates'][2][1]
            y1 = m1_relayoutData['mapbox._derived']['coordinates'][0][1]
            zoom = m1_relayoutData['mapbox.zoom']
                      
    collisions = collisions[collisions['longitude'] >= x0]
    collisions = collisions[collisions['longitude'] <= x1]
    collisions = collisions[collisions['latitude'] >= y0]
    collisions = collisions[collisions['latitude'] <= y1]   
    
    collisions2 = collisions
    if factors is not None and len(factors) != 0:
       collisions2 =collisions2[collisions2['factor'].isin(factors)]  
    if vehicle_types is not None and len(vehicle_types) != 0:
       collisions2 =collisions2[collisions2['vehicle_type'].isin(vehicle_types)]    
     
    map_fig = create_image(collisions,(x0,x1),(y0,y1), zoom = zoom)
    

    month_df = collisions['month'].value_counts()
    month_df.sort_index(inplace = True)
    month_df2 = collisions2['month'].value_counts()
    month_df2.sort_index(inplace = True)
    for i in month_df.index:
        if i in month_df2.index:
            month_df[i] = month_df[i] - month_df2[i]
    month_df = pd.DataFrame({'month':list(month_df.index), 
                            'count':list(month_df),
                            'filtering':'unselected'})
    month_df2 = pd.DataFrame({'month':list(month_df2.index), 
                            'count':list(month_df2),
                            'filtering':'selected'})
    df = month_df2.append(month_df).reset_index(drop = True)
    
    month_fig = px.bar(df, x="month", y="count", color="filtering",
                  color_discrete_map={
                    'unselected': '#636EFA',
                    'selected': '#EF553B'},
                  height=200
    )
    month_fig.update_layout(
        xaxis={#'tickangle': 35, 
            'showticklabels': True, 
            'type': 'category'},
        margin=dict(t=0,b=0,l=0,r=0)
    )        
      

    hour_df = collisions['hour'].value_counts()
    hour_df.sort_index(inplace = True)
    hour_df2 = collisions2['hour'].value_counts()
    hour_df2.sort_index(inplace = True)
    for i in hour_df.index:
        if i in hour_df2.index:
            hour_df[i] = hour_df[i] - hour_df2[i]
    hour_df = pd.DataFrame({'hour':list(hour_df.index), 
                            'count':list(hour_df),
                            'filtering':'unselected'})
    hour_df2 = pd.DataFrame({'hour':list(hour_df2.index), 
                            'count':list(hour_df2),
                            'filtering':'selected'})
    df = hour_df2.append(hour_df).reset_index(drop = True)
    
    hour_fig = px.bar(df, x="hour", y="count", color="filtering",
                  color_discrete_map={
                    'unselected': '#636EFA',
                    'selected': '#EF553B'},
                  height=200
    )
    hour_fig.update_layout(
        xaxis={#'tickangle': 35, 
            'showticklabels': True, 
            'type': 'category'},
        margin=dict(t=0,b=0,l=0,r=0)
    )
    
    factor_df = collisions['factor'].value_counts()
    factor_df.sort_index(inplace = True)
    df = pd.DataFrame({'label':list(factor_df.index), 
                        'count':list(factor_df),
                        'percent':["{:.6%}".format(x) for x in factor_df/factor_df.sum()],
                        'info':""},) 
    for i in range(len(df)):
        df['info'][i] = "<br>    " + "<br>    ".join(factor_list[df.iloc[i]['label']])
    pull_selected = [0]*len(df) 
    if factors is not None and len(factors) != 0:
        for index in df[df['label'].isin(factors)].index.values:
            pull_selected[index] = 0.2
    # if not vehicle_type == 'All':
    #    pull_selected[df[df['label']==factor].index.values[0]] = 0.2    
    factor_fig = px.pie(df, values='count', names='label',#hover_data=['percent'],
                        hover_data=['label','count','percent','info'], 
                        title = "Crash Factors - All Data")
    factor_fig.update_traces(
                      text = df['label'],
                      textinfo='text+percent',
                      textposition='inside',
                      hovertemplate='Factor: %{customdata[0][0]}<br>Count: %{customdata[0][1]}<br>Percent: %{customdata[0][2]}<br>Sub Factors: %{customdata[0][3]}',
                      pull = pull_selected
                      )
    factor_fig.update_layout(showlegend=False, 
                             title_x=0.5,
                             title_y=0.95,
                             margin=dict(t=0,b=0,l=20,r=20))
     
    vehicle_type_df = collisions['vehicle_type'].value_counts()
    vehicle_type_df.sort_index(inplace = True)
    df = pd.DataFrame({'label':list(vehicle_type_df.index), 
                        'count':list(vehicle_type_df),
                        'percent':["{:.6%}".format(x) for x in vehicle_type_df/vehicle_type_df.sum()]})
    pull_selected = [0]*len(df)
    if vehicle_types is not None and len(vehicle_types) != 0:
        for index in df[df['label'].isin(vehicle_types)].index.values:
            pull_selected[index] = 0.2
    # if not vehicle_type == 'All':
    #     pull_selected[df[df['label']==vehicle_type].index.values[0]] = 0.2
    vehicle_type_fig = px.pie(df, values='count', names='label',#hover_data=['percent'],
                              hover_data=['label','count','percent'],
                              title="Vehicle Types - All Data")
    vehicle_type_fig.update_traces(
                      text = df['label'],
                      textinfo='text+percent',
                      textposition='inside',
                      hovertemplate='Vehicle Type: %{customdata[0][0]}<br>Count: %{customdata[0][1]}<br>Percent: %{customdata[0][2]}',
                      pull = pull_selected
                      )  
    vehicle_type_fig.update_layout(showlegend=False, 
                             title_x=0.5,
                             title_y=0.95,
                             margin=dict(t=0,b=0,l=20,r=20))   
        
    map_fig2 = create_image(collisions2,(x0,x1),(y0,y1), zoom = zoom)
    
    factor_df2 =collisions2['factor'].value_counts()
    factor_df2.sort_index(inplace = True)
    df = pd.DataFrame({'label':list(factor_df2.index), 
                        'count':list(factor_df2),
                        'percent':["{:.6%}".format(x) for x in factor_df2/factor_df2.sum()],
                        'info':""},) 
    for i in range(len(df)):
        df['info'][i] = "<br>    " + "<br>    ".join(factor_list[df.iloc[i]['label']])
    factor_fig2 = px.pie(df, values='count', names='label', #hover_data=['percent'],
                         hover_data=['label','count','percent','info'],
                         title="Crash Factors - Selected Data")
    factor_fig2.update_traces(
                      text = df['label'],
                      textinfo='text+percent',
                      textposition='inside',
                      hovertemplate='Factor: %{customdata[0][0]}<br>Count: %{customdata[0][1]}<br>Percent: %{customdata[0][2]}<br>Sub Factors: %{customdata[0][3]}',
    )
    factor_fig2.update_layout(showlegend=False, 
                             title_x=0.5,
                             title_y=0.95,
                             margin=dict(t=0,b=0,l=20,r=20))
    
    vehicle_type_df2 =collisions2['vehicle_type'].value_counts()
    vehicle_type_df2.sort_index(inplace = True)
    df = pd.DataFrame({'label':list(vehicle_type_df2.index), 
                        'count':list(vehicle_type_df2),
                        'percent':["{:.6%}".format(x) for x in vehicle_type_df2/vehicle_type_df2.sum()]})
    vehicle_type_fig2 = px.pie(df, values='count', names='label', #hover_data=['percent'], 
                               hover_data=['label','count','percent'],
                               title="Vehicle Types - Selected Data")
    vehicle_type_fig2.update_traces(
                      text = df['label'],
                      textinfo='text+percent',
                      textposition='inside',
                      hovertemplate='Vehicle Type: %{customdata[0][0]}<br>Count: %{customdata[0][1]}<br>Percent: %{customdata[0][2]}',                     
    )  
    vehicle_type_fig2.update_layout(showlegend=False, 
                             title_x=0.5,
                             title_y=0.95,
                             margin=dict(t=0,b=0,l=20,r=20))
    
    total_all = "Total Collisions In The View (All Data): " + str(collisions.shape[0])
    total_filtered = "Total Collisions In The View (Selected Data): " + str(collisions2.shape[0])
        
    
    return map_fig, map_fig2, month_fig, hour_fig, factor_fig, vehicle_type_fig, factor_fig2, vehicle_type_fig2, total_all,total_filtered
  
#-----------------------------------------------------------------------------

@app.callback(
    Output('factors_dropdown', 'value'),
    [
     Input('factors_plot', 'clickData'),
     State('factors_dropdown', 'value'),
    ],
)
def update_pie_chart(clickData, dropdown_value):  
    if dropdown_value is not None: 
        current_value = dropdown_value
    else:
        current_value = []
    if clickData is not None:    
        clicked = clickData['points'][0]['label']
        if clicked in current_value:
            current_value.remove(clicked)
        else:
            current_value.append(clicked)    
    return current_value

@app.callback(
    Output('vehicle_types_dropdown', 'value'),
    [
     Input('vehicle_types_plot', 'clickData'),
     State('vehicle_types_dropdown', 'value'),
    ],
)
def update_pie_chart2(clickData, dropdown_value):
    if dropdown_value is not None: 
        current_value = dropdown_value
    else:
        current_value = []
    if clickData is not None:    
        clicked = clickData['points'][0]['label']
        if clicked in current_value:
            current_value.remove(clicked)
        else:
            current_value.append(clicked)    
    return current_value

app.run_server(debug=True)