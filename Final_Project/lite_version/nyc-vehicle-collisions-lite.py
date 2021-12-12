# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 18:35:08 2021

@author: Euclid
"""

##############################################################################
#Import required libraries
##############################################################################
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import datashader as ds, datashader.transfer_functions as tf
from functools import partial
from datashader.utils import export_image
from datashader.colors import colormap_select
from colorcet import fire
import pandas as pd
import math
#import dash_core_components as dcc
#import dash_html_components as html
#import numpy as np
#from datashader import spatial
#from datashader.utils import lnglat_to_meters as webm



##############################################################################
#Data Preparation
#
#Data Source: Motor Vehicle Collisions - Crashes (NYC Open Data)
#
#https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95
#
#In this lite version, the data is pre-processed (cleaned up) and stored in csv files
##############################################################################



#The raw data has a large number of unique factors of collision. We will load a factor table that 
#helps converting the original factors into a fewer number of revised factors
factor_cov = pd.read_csv('https://raw.githubusercontent.com/ezaccountz/Data_608/main/Final_Project/factors.csv')  
factor_cov = dict(zip(factor_cov['original factors'], factor_cov['revised factors']))

#The raw data has a large number of unique vehicle types. We will load a factor table that 
#helps converting the original vehicle types into a fewer number of revised vehicle types
vehicle_type_cov = pd.read_csv('https://raw.githubusercontent.com/ezaccountz/Data_608/main/Final_Project/vehicle_types.csv', keep_default_na=False)
vehicle_type_cov = dict(zip(vehicle_type_cov['original vehicle types'], vehicle_type_cov['revised vehicle types']))

#Distinct years, they will be the value of a dropdown list that allows the user to select a year
years = list(range(2012,2022))
years.sort(reverse=True)

#Get the list of unique revised factors
#the revised factors will be the value of a dropdown list the allows the user to select one or more factor(s)
factors = list(factor_cov.values())
factors = list(set(factors))
factors.sort()

#Create a dict, whose key is the a revised collision factor and the value is a list of original collision factors.
#A user may want to know the original factors for a selected revised factor
#We can simply load the list from this dict.
factor_list = dict()
for factor in factors:
    factor_list[factor] = []
for key, item in factor_cov.items():
    factor_list[item].append(key)

#Get the list of unique revised vehicle type
#the revised vehicle types will be the value of a dropdown list the allows the user to select one or more vehicle type(s)
vehicle_types = list(vehicle_type_cov.values())
vehicle_types = list(set(vehicle_types))
vehicle_types.sort()

#latitude and longtidue boundaries for our initial NYC map
NewYorkCity   = (( -74.25909,  -73.700181), (40.487399, 40.926178))

#setting for creating our datashader map
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

#current loaded data set (2021)
current_collisions = pd.read_csv('https://raw.githubusercontent.com/ezaccountz/Data_608/main/Final_Project/lite_version/' + str(years[0])+ '.csv')  

##############################################################################
#getDF function
#
#Try to get the dataframe for a specific year from the dfs dict.
#If the dataframe exists, return the dataframe.
#If the dataframe doesn't exist, load the data from the database, process data clean up, store the dataframe in the dfs dict and return the dataframe 
#
##############################################################################
def getDF(selected_year):
    
#In the full version, the function will be used to load the raw data from the data base and clean up the data.
#In this lite version, we simply load the pre-processed data from the csv files    
    df = pd.read_csv('https://raw.githubusercontent.com/ezaccountz/Data_608/main/Final_Project/lite_version/' + str(selected_year)+ '.csv')  
    current_collisions = df
    return df

##############################################################################
#create_image function
#
#Create a datashader map image and a plotly scatter mapbox.
#Overlay the datashader map image onto the ploty scatter mapbox
#Return the ploty scatter mapbox that can be rendered by Dash
#
##############################################################################
def create_image(df, longitude_range, latitude_range, w=plot_width, h=plot_height, zoom = 10):
    #create cvs
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=longitude_range, y_range=latitude_range)
    #Aggregation
    agg = cvs.points(df, 'longitude', 'latitude')
    #create image
    img = tf.shade(agg, cmap=fire, how='eq_hist')   
    #When the zoom variable is above a threshold, set spreading = 1 pixcel, otherwise, spreding = 0 pixel
    img = tf.spread(img, px=math.floor(zoom/10.5))
    #convert the datashader image into PIL that can be overlayed onto a ploty scatter mapbox
    img = img[::-1].to_pil()
    
    #Calculate the center of the datashader map, the single point will be used to create a ploty scatter mapbox
    coords_lat, coords_lon = agg.coords['latitude'].values, agg.coords['longitude'].values
    coordinates = [[coords_lon[0], coords_lat[0]],
               [coords_lon[-1], coords_lat[0]],
               [coords_lon[-1], coords_lat[-1]],
               [coords_lon[0], coords_lat[-1]]]  
    xcenter = [(coords_lon[0]+coords_lon[-1])/2]
    ycenter = [(coords_lat[0]+coords_lat[-1])/2]
    #Create a ploty scatter mapbox using the calculated center of the map. Set the size of the center point to 0 so it will not be showed on the image
    geo_fig = px.scatter_mapbox(lat=ycenter, 
                        lon=xcenter, 
                        size = [0],
                        width=plot_width, height=plot_height, zoom = zoom)
    #Overlay the converted datashader map image onto the ploty scatter mapbox
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
    #Return the finished ploty scatter mapbox
    return geo_fig



#Set up some html style for Dash components layout control
CONTENT_STYLE_50 = {"display": "inline-block","width": "48vw"}
CONTENT_STYLE_33 = {"display": "inline-block","width": "32vw"}
CONTENT_STYLE_66 = {"display": "inline-block","width": "65vw"}
CONTENT_STYLE_25 = {"display": "inline-block","width": "24vw"}
CONTENT_STYLE_75 = {"display": "inline-block","width": "73vw"}
CONTENT_STYLE_10 = {"display": "inline-block","width": "9vw"}
CONTENT_STYLE_90 = {"display": "inline-block","width": "88vw"}



##############################################################################
#Dash App
#
##############################################################################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server=app.server

##############################################################################
#Dash App Layout
#
##############################################################################
app.layout = html.Div([
    
    #Display a write-up/description on the visualization when the app starts, using a Bootstrap Modal 
    dbc.Modal([
          dbc.ModalHeader(dbc.ModalTitle("Project Description"),style={'height': '30px'}),
          dbc.ModalBody(html.Iframe(src='https://ezaccountz.github.io/data608/modal.html',
                                    style={'position': 'relative', 'height': '650px', 'width': '100%'}
                                    ),
                        )
    ], id="modal-body-scroll",size="xl",scrollable=False,is_open=True),
    
    html.Div([
         #Link to the write-up/description on the visualization
         html.A('Project Description',href='https://ezaccountz.github.io/data608/modal.html', target="_blank", style = {"display": "inline-block","width": "8vw"}),
        
         #Dropdown list for the user to select a year
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
        #Dropdown list for the user to select one or more collision Factor(s)
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
        #Dropdown list for the user to select one or more collision Vehicle Type(s)
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
    #A stacked bar chart the shows the number of collisions for each month through the year
    #The lower part of the bars shows the number of collisions for the selected factors and vehicle types
    #The upper part of the bars shows the number of collisions for the factors and vehicle types that are not selected
    html.Div([
         dcc.Graph(id="months_plot"),
    ],style=CONTENT_STYLE_33),
    #A stacked bar chart the shows the number of collisions for each hour through the year
    #The lower part of the bars shows the number of collisions for the selected factors and vehicle types
    #The upper part of the bars shows the number of collisions for the factors and vehicle types that are not selected
    html.Div([
         dcc.Graph(id="hours_plot"),
    ],style=CONTENT_STYLE_66),  
    #A text that shows the total number of collision ploted on the below scatter map box for all factors and vehicle types
    html.H3(id="total_all",style={'font-weight': 'bold',"text-align": "center","display": "inline-block","width": "48vw"}), 
    #A text that shows the total number of collision ploted on the below scatter map box for the selected factors and vehicle types
    html.H3(id="total_filtered",style = {'font-weight': 'bold',"text-align": "center","display": "inline-block","width": "48vw"}),  
    #A scatter map box with datashadder image overlayed for all factors and vehicle types        
    html.Div([
         dcc.Graph(id="map_plot"),
    ],style=CONTENT_STYLE_50),
    #A scatter map box with datashadder image overlayed for the selected factors and vehicle types 
    html.Div([
         dcc.Graph(id="map_plot2"),
    ],style=CONTENT_STYLE_50),
    #A pie chart that shows the percentages of the factors for the collisions within the view of the scatter map box for all factors and vehicle types
    html.Div([
         dcc.Graph(id="factors_plot"),
    ],style=CONTENT_STYLE_25),
    #A pie chart that shows the percentages of the vehicle types for the collisions within the view of the scatter map box for all factors and vehicle types
    html.Div([
         dcc.Graph(id="vehicle_types_plot"),
    ],style=CONTENT_STYLE_25),
    #A pie chart that shows the percentages of the factors for the collisions within the view of the scatter map box for the selected factors and vehicle types
    html.Div([
         dcc.Graph(id="factors_plot2"),
    ],style=CONTENT_STYLE_25),
    #A pie chart that shows the percentages of the vehicle types for the collisions within the view of the scatter map box for the selected factors and vehicle types
    html.Div([
         dcc.Graph(id="vehicle_types_plot2"),
    ],style=CONTENT_STYLE_25),  
    html.Hr(),
    html.Iframe(src='https://ezaccountz.github.io/data608/summary.html',
                                    style={'position': 'relative', 'height': '350px', 'width': '100%'}),
])

##############################################################################
#Dash App Callback Function: update_plots
#
#Inputs:
#   change in the view (location, zoom level) of the scatter map box for all factors and vehicle types  
#   change in the view (location, zoom level) of the scatter map box for the selected factors and vehicle types  
#   A different year is select from the years_dropdown
#   A factor is selected or unselected from the factors_dropdown
#   A vehicle type is selected or unselected from the vehicle_types_dropdown 
#
#Outputs:
#   re-plot the scatter map box for all factors and vehicle types, with the data points regenerated by datashader
#   re-plot the scatter map box for the selected factors and vehicle types , with the data points regenerated by datashader
#   re-plot the monthly distribution bar charts, with the new filtered data
#   re-plot the hourly distribution bar charts, with the new filtered data
#   re-plot the factors pie chart for all factors and vehicle types, with the new filtered data
#   re-plot the vehicle types pie chart for all factors and vehicle types, with the new filtered data
#   re-plot the factors pie chart for the selected factors and vehicle types, with the new filtered data
#   re-plot the vehicle types pie chart for the selected factors and vehicle types, with the new filtered data
#   regenerate the text that shows the total number of collision ploted in the scatter map box for all factors and vehicle types
#   regenerate the text that shows the total number of collision ploted in the scatter map box for the selected factors and vehicle types
#
############################################################################## 
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
def update_plots(m1_relayoutData, m2_relayoutData, year, factors, vehicle_types):
    
   
##############################################################################
# Data Preparation
##############################################################################   
    #get the name of the component that triggers the callback
    dashcc = dash.callback_context
    dashcc = dashcc.triggered[0]['prop_id'].split('.')[0]
    
    #get the dataframe for the selected year from the years_dropdown
    if dashcc == 'years_dropdown':
        collisions = getDF(year)
    else:
        collisions = current_collisions
      
    #initial coordinates and zoom level of the scatter map box
    x0 = NewYorkCity[0][0]
    x1 = NewYorkCity[0][1]
    y0 = NewYorkCity[1][0]
    y1 = NewYorkCity[1][1]   
    zoom = 8.7
    
    #If the callback is triggered by the change in the view of the scatter map box for the selected factors and vehicle types,
    #get the current coordinates of the corners and the zoom level of the map
    if dashcc == 'map_plot2':
        if m2_relayoutData is not None and 'mapbox._derived' in m2_relayoutData: 
            x0 = m2_relayoutData['mapbox._derived']['coordinates'][0][0]
            x1 = m2_relayoutData['mapbox._derived']['coordinates'][1][0]
            y0 = m2_relayoutData['mapbox._derived']['coordinates'][2][1]
            y1 = m2_relayoutData['mapbox._derived']['coordinates'][0][1]
            zoom = m2_relayoutData['mapbox.zoom']
    #Otherwise, get the current coordinates of the corners and the zoom level of the scatter map box for all factors and vehicle types
    else:
        if m1_relayoutData is not None and 'mapbox._derived' in m1_relayoutData: 
            x0 = m1_relayoutData['mapbox._derived']['coordinates'][0][0]
            x1 = m1_relayoutData['mapbox._derived']['coordinates'][1][0]
            y0 = m1_relayoutData['mapbox._derived']['coordinates'][2][1]
            y1 = m1_relayoutData['mapbox._derived']['coordinates'][0][1]
            zoom = m1_relayoutData['mapbox.zoom']
    
    #filter the data by the current boundaries of the view of the scatter maps                   
    collisions = collisions[collisions['longitude'] >= x0]
    collisions = collisions[collisions['longitude'] <= x1]
    collisions = collisions[collisions['latitude'] >= y0]
    collisions = collisions[collisions['latitude'] <= y1]   
    
    #filter the data by the selected factors and vehicle types and store the result in a new dataframe
    collisions2 = collisions
    if factors is not None and len(factors) != 0:
       collisions2 =collisions2[collisions2['factor'].isin(factors)]  
    if vehicle_types is not None and len(vehicle_types) != 0:
       collisions2 =collisions2[collisions2['vehicle_type'].isin(vehicle_types)]    
      
##############################################################################
# Scatter Map - All Data
##############################################################################               
    #create the scatter map box for all factors and vehicle types, with the data points regenerated by datashader    
    map_fig = create_image(collisions,(x0,x1),(y0,y1), zoom = zoom)
    
##############################################################################
# Scatter Map - Selected Data (selected collision factors and vehicle types)
##############################################################################     
    #create the scatter map box for the selected factors and vehicle types, with the data points regenerated by datashader    
    map_fig2 = create_image(collisions2,(x0,x1),(y0,y1), zoom = zoom)

##############################################################################
# Bar Chart - Monthly Distribution
##############################################################################         
    #Calculate the number of collisions for each month for all factors and vehicle types
    month_df = collisions['month'].value_counts()
    month_df.sort_index(inplace = True)
    #Calculate the number of collisions for each month for the selected factors and vehicle types
    month_df2 = collisions2['month'].value_counts() 
    month_df2.sort_index(inplace = True)
    #Calculate the number of collisions for each month for the factors and vehicle types that are not selected
    for i in month_df.index:
        if i in month_df2.index:
            month_df[i] = month_df[i] - month_df2[i]
    #Combine the data for the selected factors and vehicle types and the data for the factors and vehicle types that are not selected into one dataframe for plotting
    month_df = pd.DataFrame({'month':list(month_df.index), 
                            'count':list(month_df),
                            'filtering':'unselected'})
    month_df2 = pd.DataFrame({'month':list(month_df2.index), 
                            'count':list(month_df2),
                            'filtering':'selected'})
    df = month_df2.append(month_df).reset_index(drop = True)
    #Create a stacked bar chart the shows the number of collisions for each month through the year
    #The lower part of the bars shows the number of collisions for the selected factors and vehicle types
    #The upper part of the bars shows the number of collisions for the factors and vehicle types that are not selected
    month_fig = px.bar(df, x="month", y="count", color="filtering",
                  color_discrete_map={
                    'unselected': '#636EFA',
                    'selected': '#EF553B'},
                  height=200
    )
    month_fig.update_layout(
        xaxis={
            'showticklabels': True, 
            'type': 'category'},
        margin=dict(t=0,b=0,l=0,r=0)
    )        
     
##############################################################################
# Bar Chart - Hourly Distribution
##############################################################################    
    #Calculate the number of collisions for each hour for all factors and vehicle types
    hour_df = collisions['hour'].value_counts()
    hour_df.sort_index(inplace = True)
    #Calculate the number of collisions for each hour for the selected factors and vehicle types
    hour_df2 = collisions2['hour'].value_counts()
    hour_df2.sort_index(inplace = True)
    #Calculate the number of collisions for each hour for the factors and vehicle types that are not selected
    for i in hour_df.index:
        if i in hour_df2.index:
            hour_df[i] = hour_df[i] - hour_df2[i]
    #Combine the data for the selected factors and vehicle types and the data for the factors and vehicle types that are not selected into one dataframe for plotting
    hour_df = pd.DataFrame({'hour':list(hour_df.index), 
                            'count':list(hour_df),
                            'filtering':'unselected'})
    hour_df2 = pd.DataFrame({'hour':list(hour_df2.index), 
                            'count':list(hour_df2),
                            'filtering':'selected'})
    df = hour_df2.append(hour_df).reset_index(drop = True)
    #Create a stacked bar chart the shows the number of collisions for each hour through the year
    #The lower part of the bars shows the number of collisions for the selected factors and vehicle types
    #The upper part of the bars shows the number of collisions for the factors and vehicle types that are not selected
    hour_fig = px.bar(df, x="hour", y="count", color="filtering",
                  color_discrete_map={
                    'unselected': '#636EFA',
                    'selected': '#EF553B'},
                  height=200
    )
    hour_fig.update_layout(
        xaxis={
            'showticklabels': True, 
            'type': 'category'},
        margin=dict(t=0,b=0,l=0,r=0)
    )

##############################################################################
# Factors Pie Chart - All Data
##############################################################################    
    #Calculate the number of collisions for each factor for all factors and vehicle types
    factor_df = collisions['factor'].value_counts()
    factor_df.sort_index(inplace = True)
    df = pd.DataFrame({'label':list(factor_df.index), 
                        'count':list(factor_df),
                        'percent':["{:.6%}".format(x) for x in factor_df/factor_df.sum()],
                        'info':""},) 
    #For each factor that will be plotted in the pie chart, gather the names of the original factors and combine them into one string, separated by <br>
    for i in range(len(df)):
        df['info'][i] = "<br>    " + "<br>    ".join(factor_list[df.iloc[i]['label']])
    #Find the indexs of the factors selected in the factors dropdown, set the pull values to 0.2 so that the factors are showing as pulled on the pie chart
    pull_selected = [0]*len(df) 
    if factors is not None and len(factors) != 0:
        for index in df[df['label'].isin(factors)].index.values:
            pull_selected[index] = 0.2 
    #Create a pie chart that shows the percentages of the factors for the collisions within the view of the scatter map box for all factors and vehicle types
    factor_fig = px.pie(df, values='count', names='label',
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

##############################################################################
# Vehicle Types Pie Chart - All Data
##############################################################################     
    #Calculate the number of collisions for each vehicle type for all factors and vehicle types
    vehicle_type_df = collisions['vehicle_type'].value_counts()
    vehicle_type_df.sort_index(inplace = True)
    df = pd.DataFrame({'label':list(vehicle_type_df.index), 
                        'count':list(vehicle_type_df),
                        'percent':["{:.6%}".format(x) for x in vehicle_type_df/vehicle_type_df.sum()]})
    #For each vehicle type that will be plotted in the pie chart, gather the names of the original vehicle types and combine them into one string, separated by <br>
    pull_selected = [0]*len(df)
    if vehicle_types is not None and len(vehicle_types) != 0:
        for index in df[df['label'].isin(vehicle_types)].index.values:
            pull_selected[index] = 0.2
    #Create a pie chart that shows the percentages of the vehicle types for the collisions within the view of the scatter map box for all factors and vehicle types
    vehicle_type_fig = px.pie(df, values='count', names='label',
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
        
##############################################################################
# Factors Pie Chart - Selected Data (selected collision factors and vehicle types)
##############################################################################    
    #Calculate the number of collisions for each factor for the selected factors and vehicle types
    factor_df2 =collisions2['factor'].value_counts()
    factor_df2.sort_index(inplace = True)
    df = pd.DataFrame({'label':list(factor_df2.index), 
                        'count':list(factor_df2),
                        'percent':["{:.6%}".format(x) for x in factor_df2/factor_df2.sum()],
                        'info':""},) 
    #For each factor that will be plotted in the pie chart, gather the names of the original factors and combine them into one string, separated by <br>
    for i in range(len(df)):
        df['info'][i] = "<br>    " + "<br>    ".join(factor_list[df.iloc[i]['label']])
    #Create a pie chart that shows the percentages of the factors for the collisions within the view of the scatter map box for the selected factors and vehicle types
    factor_fig2 = px.pie(df, values='count', names='label', 
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
    
##############################################################################
# Vehicle Types Pie Chart - Selected Data (selected collision factors and vehicle types)
##############################################################################  
    #Calculate the number of collisions for each vehicle type for the selected factors and vehicle types
    vehicle_type_df2 =collisions2['vehicle_type'].value_counts()
    vehicle_type_df2.sort_index(inplace = True)
    df = pd.DataFrame({'label':list(vehicle_type_df2.index), 
                        'count':list(vehicle_type_df2),
                        'percent':["{:.6%}".format(x) for x in vehicle_type_df2/vehicle_type_df2.sum()]})
    #Create pie chart that shows the percentages of the vehicle types for the collisions within the view of the scatter map box for the selected factors and vehicle types
    vehicle_type_fig2 = px.pie(df, values='count', names='label', 
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
    
    #generate the text that shows the total number of collision ploted in the scatter map box for all factors and vehicle types
    total_all = "Total Collisions In The View (All Data): " + str(collisions.shape[0])
    #generate the text that shows the total number of collision ploted in the scatter map box for the selected factors and vehicle types
    total_filtered = "Total Collisions In The View (Selected Data): " + str(collisions2.shape[0])
       
    return map_fig, map_fig2, month_fig, hour_fig, factor_fig, vehicle_type_fig, factor_fig2, vehicle_type_fig2, total_all, total_filtered
  
    
##############################################################################
#Dash App Callback Function: factor_pie_chart
#
#Inputs:
#   A click on a factor on the pie chart that shows the percentages of the factors for the collisions within the view of the scatter map box for all factors and vehicle types
#
#Outputs:
#   If the clicked factor is already selected in the factors dropdown,, unselect the factor, which will trigger the update_plots callback function
#   If the clicked factor is not selected in the factors dropdown,, select the factor, which will trigger the update_plots callback function
#
############################################################################## 
@app.callback(
    Output('factors_dropdown', 'value'),
    [
     Input('factors_plot', 'clickData'),
     State('factors_dropdown', 'value'),
    ],
)
def factor_pie_chart(clickData, dropdown_value):  
    #get the current values of the factors dropdown
    if dropdown_value is not None: 
        current_value = dropdown_value
    else:
        current_value = []
    if clickData is not None:  
        #get the name of the factor that is clicked on
        clicked = clickData['points'][0]['label']
        #If the factor is already selected in the factors dropdown, unselect it
        if clicked in current_value:
            current_value.remove(clicked)
        #If the factor is not selected in the factors dropdown, select it
        else:
            current_value.append(clicked)    
    return current_value

##############################################################################
#Dash App Callback Function: vehicle_type_pie_chart
#
#Inputs:
#   A click on a vehicle type on the pie chart that shows the percentages of the vehicle types for the collisions within the view of the scatter map box for all factors and vehicle types
#
#Outputs:
#   If the clicked vehicle type is already selected in the factors dropdown,, unselect the vehicle type, which will trigger the update_plots callback function
#   If the clicked vehicle type is not selected in the vehicle types dropdown,, select the vehicle type, which will trigger the update_plots callback function
#
############################################################################## 
@app.callback(
    Output('vehicle_types_dropdown', 'value'),
    [
     Input('vehicle_types_plot', 'clickData'),
     State('vehicle_types_dropdown', 'value'),
    ],
)
def vehicle_type_pie_chart(clickData, dropdown_value):
    #get the current values of the vehicle types dropdown
    if dropdown_value is not None: 
        current_value = dropdown_value
    else:
        current_value = []
    #get the name of the vehicle type that is clicked on
    if clickData is not None:    
        clicked = clickData['points'][0]['label']
        #If the vehicle type is already selected in the vehicle types dropdown, unselect it
        if clicked in current_value:
            current_value.remove(clicked)
        #If the vehicle type is not selected in the vehicle types dropdown, select it
        else:
            current_value.append(clicked)    
    return current_value

if __name__ == '__main__':
    app.run_server(debug=True)