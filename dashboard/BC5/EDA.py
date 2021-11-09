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

# https://htmlcheatsheet.com/css/

######################################################Data##############################################################
#df = pd.read_csv("C:/Users/Pedro/Desktop/Business Cases/BC5/Datasets/eda_dash.csv")
df = pd.read_csv('C:/Users/migue/Desktop/Datasets/eda_dash.csv')
#df = pd.read_csv("C:/Users/bruno/OneDrive/Ambiente de Trabalho/Datasets/eda_dash.csv")

df["Year"] = df["Date"].str.split("-").str[0]
df['Date'] = pd.to_datetime(df['Date'])

######################################################Interactive Components############################################

points_of_sale = ["1", "2", "3", "4", '5',   '6',   '7',   '8',   '9',  '10',  '11',  '12',  '13',
        '14',  '15',  '16',  '17',  '18',  '19',  '20']

pos_options = [dict(label='' + pos, value=pos) for pos in points_of_sale]

pos_dropdown = dcc.Dropdown(
    id='pos_drop',
    options=pos_options,
    value='1',
    persistence=True,
    persistence_type='session'
)

quarters = ["1", "2", "3", "4"]

quarters_options = [dict(label='' + quarter, value=quarter) for quarter in quarters]

quarters_dropdown = dcc.Dropdown(
    id='quarters_drop',
    options=quarters_options,
    value='1',
    persistence=True,
    persistence_type='session'
)

years = ["2016", "2017", "2018", "2019"]

years_options = [dict(label='' + year, value=year) for year in years]

years_dropdown = dcc.Dropdown(
    id='years_drop',
    options=years_options,
    value='2016',
    persistence=True,
    persistence_type='session'
)
options = ["Units", "Value"]

options_choose = [dict(label='' + opt, value=opt) for opt in options]

options_dropdown = dcc.Dropdown(
    id='options_drop',
    options=options_choose,
    value='Units',
    persistence=True,
    persistence_type='session'
)

##################################################APP###################################################################

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4('Select the PoS:', className="text-center")
        ], width=3),
        dbc.Col([
            html.H4('Select the quarter:', className="text-center")
        ], width=3),
        dbc.Col([
            html.H4('Select the year:', className="text-center")
        ], width=3),
        dbc.Col([
            html.H4('Select the option:', className="text-center")
        ], width=3)
    ]),
    dbc.Row([
        dbc.Col([
            pos_dropdown,
        ], width=3),
        dbc.Col([
            quarters_dropdown,
        ], width=3),
        dbc.Col([
            years_dropdown,
        ], width=3),
        dbc.Col([
            options_dropdown,
        ], width=3)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dcc.Graph(id='graph_value', style={'height': 500}),
                body=True, color="#31343b"
            )
        ], width={'size': 6}),
        dbc.Col([
            dbc.Card(
                dcc.Graph(id='graph_prod_fam', style={'height': 500}),
                body=True, color="#31343b"
            )
        ], width={'size': 6})
    ], className="my-2"),
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dcc.Graph(id='graph_market_share', style={'height': 500}),
                body=True, color="#31343b"
            )
        ], width={'size': 6}),
        dbc.Col([
            dbc.Card(
                dcc.Graph(id='graph_market_share2', style={'height': 500}),
                body=True, color="#31343b"
            )
        ], width={'size': 6})
    ], className="my-2"),

], fluid=True)


@app.callback(
    Output('graph_value', 'figure'),
    [Input('pos_drop', 'value'),
     Input('quarters_drop', 'value'),
     Input('years_drop', 'value'),
     Input('options_drop', 'value')]
)
def graph_1(pos, quarter, year, option):
    df_1 = df.loc[df['Point-of-Sale_ID'] == int(pos)]
    df_1 = df_1[(df_1['Quarter'] == int(quarter)) & (df_1["Year"] == year)]
    df_1 = df_1.groupby(df_1['Date']).sum()

    fig = px.bar(df_1, x=df_1.index, y=df_1[option], title=option + " for Quarter " + quarter + " and Year " + year
                                                           + " in Point of Sale " + pos)
    fig.update_layout(paper_bgcolor='rgba(255,255,255)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Date",
                       yaxis_title=option)
    fig.update_traces(
        hovertemplate='Date: %{x} <br>'+option+' Sold: %{y} <extra></extra>')
    return fig


@app.callback(
    Output('graph_prod_fam', 'figure'),
    [Input('pos_drop', 'value'),
     Input('quarters_drop', 'value'),
     Input('years_drop', 'value'),
     Input('options_drop', 'value')]
)
def graph_2(pos, quarter, year, option):
    df_2 = df.loc[df['Point-of-Sale_ID'] == int(pos)]
    df_2 = df_2[(df_2['Quarter'] == int(quarter)) & (df_2["Year"] == year)]

    df_2 = df_2.groupby(df['ProductFamily_ID']).sum()
    df_2 = df_2.sort_values(by="Units", ascending=False)

    df_2 = df_2.head(10)


    fig2 = px.bar(df_2, x=df_2.index, y=df_2[option], title="ProductFamilyID" + " for Quarter " + quarter + " and Year " + year
                                                           + " in Point of Sale " + pos)
    fig2.update_layout(paper_bgcolor='rgba(255,255,255)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="ProductFamilyID",
    yaxis_title=option)
    fig2.update_traces(
        hovertemplate='ProductFamilyID: %{x} <br>'+option+' Sold: %{y} <extra></extra>')
    return fig2

@app.callback(
    Output('graph_market_share', 'figure'),
    [Input('pos_drop', 'value'),
     Input('quarters_drop', 'value'),
     Input('years_drop', 'value'),
     Input('options_drop', 'value')]
)
def graph_3(pos, quarter, year, option):
    df_3 = df.loc[df['Point-of-Sale_ID'] == int(pos)]
    df_3 = df_3[(df_3['Quarter'] == int(quarter)) & (df_3["Year"] == year)]

    df_3 = df_3.groupby(df['ProductFamily_ID']).sum()
    df_3 = df_3.sort_values(by=option, ascending=False).head(10)

    fig3 = px.pie(df_3, values=df_3[option], names=df_3.index, title='Top 10 - Market Share for ProductFamilyID')

    return fig3



@app.callback(
    Output('graph_market_share2', 'figure'),
    [Input('pos_drop', 'value'),
     Input('quarters_drop', 'value'),
     Input('years_drop', 'value'),
     Input('options_drop', 'value')]
)
def graph_4(pos, quarter, year, option):
    df_4 = df.loc[df['Point-of-Sale_ID'] == int(pos)]
    df_4 = df_4[(df_4['Quarter'] == int(quarter)) & (df_4["Year"] == year)]

    df_4 = df_4.groupby(df['ProductCategory_ID']).sum()
    df_4 = df_4.sort_values(by=option, ascending=False).head(10)

    fig4 = px.pie(df_4, values=df_4[option], names=df_4.index,
                  title='Top 10 - Market Share for ProductCategory_ID')
    return fig4



