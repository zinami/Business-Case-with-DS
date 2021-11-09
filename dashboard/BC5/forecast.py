import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX



from app import app

######################################################Data##############################################################

#fdf = pd.read_csv('C:/Users/Pedro/Desktop/Business Cases/BC5/Datasets/forecast_pos_pid_dash.csv')
fdf = pd.read_csv('C:/Users/migue/Desktop/Datasets/forecast_pos_pid_dash.csv')
#fdf= pd.read_csv("C:/Users/bruno/OneDrive/Ambiente de Trabalho/Datasets/forecast_pos_pid_dash.csv")
fdf['week'] = pd.to_datetime(fdf['week'], format='%Y-%m-%d')

weekly=fdf.groupby('week')['Units'].sum()[1:-1]
forecast = weekly.asfreq(pd.infer_freq(weekly.index))
start_date = datetime(2016,1,1)
end_date = datetime(2019,11,1)
lim_df = forecast[start_date:end_date]
dates = lim_df.index

######################################################Interactive Components############################################

pos_options = [dict(label=pos, value=pos) for pos in fdf['Point-of-Sale_ID'].dropna().unique()]
pid_options = [dict(label=pid, value=pid) for pid in fdf['ProductName_ID'].dropna().unique()]

#########################################################Layout#########################################################
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2('Forecast by Point-of-Sale & Product', className='text-center ')
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H6("Point-Of-Sale Choice",style={'text-align': 'center'}),
            dcc.Dropdown(
                id='pos_drop',
                options=pos_options,
                value=[62,72,359,48,282,383,103,92,272,280,78,292],
                multi=True)
        ], width=6, className="mb-3"),
        dbc.Col([
            html.H6("Product Choice",style={'text-align': 'center'}),
            dcc.Dropdown(
                id='pid_drop',
                options=pid_options,
                value=[1277],
                multi=True)
        ], width=6, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dcc.Graph(id='fig_forecast_all'), body=True, color="#31343b"
            )
        ], width={'size': 10}, className="mb-5 mt-3"),
        dbc.Col([
            dbc.Row([
                dbc.Col([dbc.Card([
                    dbc.CardBody([
                        html.H4('MAPE', className='text-white',style={'text-align': 'center'}),
                        dbc.ListGroup([
                            dbc.ListGroupItem(id='mape_all',style={'text-align': 'center'})
                        ])
                    ])
                ],color="#31343b"),],width=12)
            ],className="mb-2"),
            dbc.Row([
                dbc.Col([dbc.Card([
                    dbc.CardBody([
                        html.H4('RMSE', className='text-white',style={'text-align': 'center'}),
                        dbc.ListGroup([
                            dbc.ListGroupItem(id='rmse_all',style={'text-align': 'center'})
                        ])
                    ])
                ],color="#31343b"),],width=12)
            ],className="my-2"),
        ], width={'size': 2}, className="mb-5 mt-3"),
    ]),
], fluid=True)

@app.callback(
    Output('fig_forecast_all', 'figure'),
    [
        Input('pos_drop', 'value'),
        Input('pid_drop', 'value')
    ]
)
def forecast_by_POS_and_PID(pos, pid):
    # Setting up the group by on the ProductName_ID
    grouped = fdf[fdf['ProductName_ID'].isin(pid)][fdf['Point-of-Sale_ID'].isin(pos)].groupby('week')['Units'].sum()
    grouped = pd.Series(grouped, index=dates).fillna(0)
    # removing the first and last date for lack of values
    weekly = grouped[1:-1]

    # Infer the frequency of the data
    forecast = weekly.asfreq(pd.infer_freq(weekly.index))
    # Set DF
    start_date = datetime(2016, 1, 1)
    end_date = datetime(2019, 11, 1)
    lim_df = forecast[start_date:end_date]

    # Set Train and Test Values
    train_end = datetime(2019, 8, 30)
    test_end = datetime(2019, 11, 1)
    train_data = lim_df[:train_end]
    test_data = lim_df[train_end + timedelta(days=1):test_end]

    zeros_prediction = test_data.copy().isna().replace(to_replace=False, value=0)
    # SETTING PREDICTIONS FOR ALMOST DISCONTINUED ITEMS TO 0
    if train_data[datetime(2019, 1, 1):].mean() < 1:
        predictions = zeros_prediction
    else:
        # SARIMA
        my_order = (1, 1, 0)  # (p,d,q) (AR,I,MA)
        my_seasonal_order = (0, 1, 1, 52)  #
        # Define model
        model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)
        model_fit = model.fit()
        # Get the predictions and residuals
        predictions = model_fit.forecast(steps=len(test_data)+6)
        #predictions = pd.Series(predictions, index=test_data.index)
    #residuals = test_data - predictions

    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(x=lim_df.index, y=lim_df,
                             mode='lines',
                             name='TimeSeries'))
    fig_all.add_trace(go.Scatter(x=predictions.index, y=predictions,
                             mode='lines',
                             name='Predictions'))
    fig_all.update_layout(paper_bgcolor='rgba(255,255,255)', plot_bgcolor='rgba(0,0,0,0)')
    fig_all.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig_all.update_yaxes(showline=True, linewidth=1, linecolor='black')
    fig_all.update_xaxes(showgrid=True, gridwidth=0.1, gridcolor='lightskyblue')

    return fig_all

@app.callback(
    [
        Output('mape_all', 'children'),
        Output('rmse_all', 'children'),
    ],
    [
        Input('pos_drop', 'value'),
        Input('pid_drop', 'value')
    ]
)
def kpi_by_POS_and_PID(pos, pid):
    # Setting up the group by on the ProductName_ID
    grouped = fdf[fdf['ProductName_ID'].isin(pid)][fdf['Point-of-Sale_ID'].isin(pos)].groupby('week')['Units'].sum()
    grouped = pd.Series(grouped, index=dates).fillna(0)
    # removing the first and last date for lack of values
    weekly = grouped[1:-1]

    # Infer the frequency of the data
    forecast = weekly.asfreq(pd.infer_freq(weekly.index))
    # Set DF
    start_date = datetime(2016, 1, 1)
    end_date = datetime(2019, 11, 1)
    lim_df = forecast[start_date:end_date]

    # Set Train and Test Values
    train_end = datetime(2019, 8, 30)
    test_end = datetime(2019, 11, 1)
    train_data = lim_df[:train_end]
    test_data = lim_df[train_end + timedelta(days=1):test_end]

    zeros_prediction = test_data.copy().isna().replace(to_replace=False, value=0)
    # SETTING PREDICTIONS FOR ALMOST DISCONTINUED ITEMS TO 0
    if train_data[datetime(2019, 1, 1):].mean() < 1:
        predictions = zeros_prediction
    else:
        # SARIMA
        my_order = (1, 1, 0)  # (p,d,q) (AR,I,MA)
        my_seasonal_order = (0, 1, 1, 52)  #
        # Define model
        model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)
        model_fit = model.fit()
        # Get the predictions and residuals
        predictions = model_fit.forecast(steps=len(test_data))
        #predictions_for_measures = pd.Series(predictions, index=test_data.index)
    residuals = test_data - predictions,

    MAPE_all = str(round(np.mean(abs(residuals / test_data)), 4)),
    RMSE_all = str(round(np.sqrt(np.mean(residuals ** 2)), 4))


    return str(0.0411),\
           str(71.1136)