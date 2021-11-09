import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px

from app import app
from app import server
import EDA
#df = pd.read_csv("C:/Users/bruno/OneDrive/Ambiente de Trabalho/Datasets/pca_product.csv")
#product = pd.read_csv("C:/Users/bruno/OneDrive/Ambiente de Trabalho/Datasets/final_product.csv")
df = pd.read_csv("C:/Users/migue/Desktop/Datasets/pca_product.csv")
product = pd.read_csv("C:/Users/migue/Desktop/Datasets/final_product.csv")

product.set_index('Point-of-Sale_ID',inplace=True)
df.set_index('Point-of-Sale_ID',inplace=True)

def scatter_plot_product():
    fig = px.scatter(df, x="PC0", y="PC1", color="cluster_product", hover_data=[df.index])
    fig.update_layout(paper_bgcolor='rgba(255,255,255)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def count_label(df,label,columnToCount):
    d ={'Cluster':df.groupby(label)[columnToCount].count().index,'Count':df.groupby(label)[columnToCount].count().values}
    df =pd.DataFrame(data=d )
    fig = px.bar(df, x='Cluster', y='Count')
    fig.update_layout(paper_bgcolor='rgba(255,255,255)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_top10(clusterNumber):
    d ={'Products':product[product['cluster_product']==clusterNumber].mean().sort_values(ascending=False).head(10).index,'Mean Sold':product[product['cluster_product']==clusterNumber].mean().sort_values(ascending=False).head(10).values}
    df =pd.DataFrame(data=d )
    if clusterNumber == 0:
        fig = px.bar(df, y='Products', x='Mean Sold',title = 'Top 10 products from cluster 0 - High Value',orientation='h')
        fig.update_layout(paper_bgcolor='rgba(255,255,255)', plot_bgcolor='rgba(0,0,0,0)')
    elif clusterNumber == 1:
        fig = px.bar(df, y='Products', x='Mean Sold',title = 'Top 10 products from cluster 1 - Medium Value',orientation='h')
        fig.update_layout(paper_bgcolor='rgba(255,255,255)', plot_bgcolor='rgba(0,0,0,0)')
    else:
        fig = px.bar(df, y='Products', x='Mean Sold',title = 'Top 10 products from cluster 2 - Low Value',orientation='h')
        fig.update_layout(paper_bgcolor='rgba(255,255,255)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dcc.Graph(figure=scatter_plot_product(), style={'height': 500}),
                body=True, color="#31343b"
            )
        ], width={'size': 6}),

        dbc.Col([
            dbc.Card(
                dcc.Graph(figure=count_label(df, 'cluster_product', 'PC0'), style={'height': 500}),
                body=True, color="#31343b"
            )
        ], width={'size': 6})
    ],className="mb-2"),

    dbc.Row([
        dbc.Col([
            dbc.Card(
                dcc.Graph(figure=plot_top10(0), style={'height': 500}),
                body=True, color="#31343b"
            )
        ], width={'size': 4}),
        dbc.Col([
            dbc.Card(
                dcc.Graph(figure=plot_top10(1), style={'height': 500}),
                body=True, color="#31343b"
            )
        ], width={'size': 4}),
        dbc.Col([
            dbc.Card(
                dcc.Graph(figure=plot_top10(2), style={'height': 500}),
                body=True, color="#31343b"
            )
        ], width={'size': 4}),
    ])
], fluid=True)