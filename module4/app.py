# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 15:09:38 2021

@author: Euclid
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import plotly.express as px

app_token = 'O8FhYuPxrY1stJIoIJ21P4WUa'

address = 'http://data.cityofnewyork.us/resource/uvpi-gqnh.json'
select_columns = ['spc_common','boroname','steward','health','count(health)']
group_columns = ['spc_common','boroname','steward','health']
limit = 9999

url = address
url = url + "?$select="+",".join(select_columns)
url = url + "&$group="+",".join(group_columns)
url = url + "&$limit="+str(limit)
url = url + "&$$app_token="+app_token

trees = pd.read_json(url)
trees.rename(columns={'count_health':'count'},inplace=True)
trees.dropna(inplace=True)
trees.replace("None","0(None)",inplace=True)

species = trees['spc_common'].unique()
species = np.insert(species,0,'All')
boros = trees['boroname'].unique()
steward = trees['steward'].unique()
health = trees['health'].unique()

def q1_plot(df,boro,specie):
    if specie != 'All':
        df = df[df['spc_common'] == specie]
    if boro != 'All':
        df = df[df['boroname'] == boro]
    df = df.groupby(['health'],as_index=False).sum()
    total = sum(df['count'])
    df['percent'] = df['count']/total   
    fig = px.bar(df, x="health", y="percent", hover_data=['count', 'percent'])    
    if (len(df) == 0):
        fig.update_layout(yaxis={'visible': False, 'showticklabels': False},
                          xaxis={'visible': False, 'showticklabels': False},
                          title={
                            'text': "No Data for " + boro,
                            'y':0.5,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'}
                         )                         
    else: 
        fig.update_layout(
        width=400,
        height=300,
        title={
            'text': boro,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    return fig


def q2_plot(df,boro,specie):
    if specie != 'All':
        df = df[df['spc_common'] == specie] 
    if boro != 'All':
        df = df[df['boroname'] == boro]
        
    df = df.groupby(['health','steward'],as_index=False).sum()
    df = df.pivot(index='health', columns='steward', values='count')
    df.fillna(0, inplace=True)
    for sted in df.columns: 
        df[sted] = df[sted]/sum(df[sted])
    fig = px.imshow(df,color_continuous_scale="Teal",labels=dict(color="percentage"))        
    if (len(df) == 0):
        fig.update_layout(yaxis={'visible': False, 'showticklabels': False},
                          xaxis={'visible': False, 'showticklabels': False},
                          title={
                            'text': "No Data for " + boro,
                            'y':0.5,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'}
                         )                         
    else: 
        fig.update_layout(
        width=400,
        height=300,
        title={
            'text': boro,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    return fig

SIDEBAR_STYLE = {
    "display": "inline-block",
    "width": "15vw",
    "background-color": "#f8f9fa",
    "vertical-align": "top"
}

CONTENT_STYLE_1 = {
    "display": "inline-block",
    "width": "27vw"
}

CONTENT_STYLE_2 = {
    "display": "inline-block",
    "width": "27vw"
}

CONTENT_STYLE_3 = {
    "display": "inline-block",
    "width": "27vw"
}

#-----------------------------------------------------------------------------

app = dash.Dash(__name__,suppress_callback_exceptions=True)
server=app.server

app.layout = html.Div([
    dcc.Tabs(id="data_608_module4", value='tab_1', children=[
        dcc.Tab(label='Q1:Proportion of trees in good, fair, or poor health condition', value='tab_1'),
        dcc.Tab(label='Q2:Stewards vs. health conditions', value='tab_2'),
    ]),
    html.Div(id='tabs_content')
])

@app.callback(Output('tabs_content', 'children'),
              Input('data_608_module4', 'value'))
def render_content(tab):
    if tab == 'tab_1':
        return html.Div([
            html.Div([
                html.H4("Species",style={'font-weight': 'bold',"text-align": "center"}),
                dcc.Dropdown(
                    id="q1_dropdown",
                    options=[{"label": x, "value": x} for x in species],
                    value=species[0],
                    clearable=False,
                )
            ],style=SIDEBAR_STYLE),
            html.Div([
                dcc.Graph(id="q1_chart1"),
                dcc.Graph(id="q1_chart4"),
            ],style=CONTENT_STYLE_1),
            html.Div([
                dcc.Graph(id="q1_chart2"),
                dcc.Graph(id="q1_chart5"),
            ],style=CONTENT_STYLE_2),
            html.Div([
                dcc.Graph(id="q1_chart3"),
                dcc.Graph(id="q1_chart6"),
            ],style=CONTENT_STYLE_3),
        ])
    elif tab == 'tab_2':
        return html.Div([
            html.Div([
                html.H4("Species",style={'font-weight': 'bold',"text-align": "center"}),
                dcc.Dropdown(
                    id="q2_dropdown",
                    options=[{"label": x, "value": x} for x in species],
                    value=species[0],
                    clearable=False,
                )
            ],style=SIDEBAR_STYLE),
            html.Div([
                dcc.Graph(id="q2_chart1"),
                dcc.Graph(id="q2_chart4"),
            ],style=CONTENT_STYLE_1),
            html.Div([
                dcc.Graph(id="q2_chart2"),
                dcc.Graph(id="q2_chart5"),
            ],style=CONTENT_STYLE_2),
            html.Div([
                dcc.Graph(id="q2_chart3"),
                dcc.Graph(id="q2_chart6"),
            ],style=CONTENT_STYLE_3),
        ])
    
#-----------------------------------------------------------------------------

@app.callback(
    Output("q1_chart1", "figure"), Output("q1_chart2", "figure"),
    Output("q1_chart3", "figure"), Output("q1_chart4", "figure"),
    Output("q1_chart5", "figure"), Output("q1_chart6", "figure"),
    [Input("q1_dropdown", "value")])
def q1_update_bar_chart(specie):      
    fig1 = q1_plot(trees, 'All',specie) 
    fig2 = q1_plot(trees, boros[0],specie) 
    fig3 = q1_plot(trees, boros[1],specie) 
    fig4 = q1_plot(trees, boros[2],specie) 
    fig5 = q1_plot(trees, boros[3],specie) 
    fig6 = q1_plot(trees, boros[4],specie) 
    return fig1, fig2, fig3, fig4, fig5, fig6

@app.callback(
    Output("q2_chart1", "figure"), Output("q2_chart2", "figure"),
    Output("q2_chart3", "figure"), Output("q2_chart4", "figure"),
    Output("q2_chart5", "figure"), Output("q2_chart6", "figure"),
    [Input("q2_dropdown", "value")])
def q2_update_bar_chart(specie):
    fig1 = q2_plot(trees, 'All',specie) 
    fig2 = q2_plot(trees, boros[0],specie) 
    fig3 = q2_plot(trees, boros[1],specie) 
    fig4 = q2_plot(trees, boros[2],specie) 
    fig5 = q2_plot(trees, boros[3],specie) 
    fig6 = q2_plot(trees, boros[4],specie) 
    return fig1, fig2, fig3, fig4, fig5, fig6


if __name__ == '__main__':
    app.run_server(debug=True)
