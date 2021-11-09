import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX



from app import app
from app import server
import EDA

######################################################Data##############################################################
#pdf = pd.read_csv('C:/Users/Pedro/Desktop/Business Cases/BC5/Datasets/product_df.csv')
pdf = pd.read_csv('C:/Users/migue/Desktop/Datasets/product_df.csv')
#pdf= pd.read_csv("C:/Users/bruno/OneDrive/Ambiente de Trabalho/Datasets/product_df.csv")
pdf['week'] = pd.to_datetime(pdf['week'], format='%Y-%m-%d')

weekly=pdf.groupby('week')['Units'].sum()[1:-1]
forecast = weekly.asfreq(pd.infer_freq(weekly.index))
start_date = datetime(2016,1,1)
end_date = datetime(2019,11,1)
lim_df = forecast[start_date:end_date]
dates = lim_df.index

######################################################Interactive Components############################################
pid_options = [dict(label=pid, value=pid) for pid in pdf['ProductName_ID'].dropna().unique()]


######################################################Layout############################################
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2('Forecast by Product Name', className='text-center ')
        ], width=12)
    ]),
    dbc.Row(
        dbc.Col([
            html.H6("Product Name Choice", className='text-left text mb-4 ml-4'),
            dcc.Dropdown(
                id='pid_drop',
                options=pid_options,
                value=[2609],
                multi=True)
        ], width=12, className="mb-3")
    ),
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dcc.Graph(id='prod_forecast_callback'), body=True, color="#31343b"
            )
        ], width={'size': 10}, className="mb-5 mt-3"),
        dbc.Col([
            dbc.Row([
                dbc.Col([dbc.Card([
                    dbc.CardBody([
                        html.H4('MAPE', className='text-white',style={'text-align': 'center'}),
                        dbc.ListGroup([
                            dbc.ListGroupItem(id='mape_prod',style={'text-align': 'center'})
                        ])
                    ])
                ],color="#31343b"),],width=12)
            ],className="mb-2"),
            dbc.Row([
                dbc.Col([dbc.Card([
                    dbc.CardBody([
                        html.H4('RMSE', className='text-white',style={'text-align': 'center'}),
                        dbc.ListGroup([
                            dbc.ListGroupItem(id='rmse_prod',style={'text-align': 'center'})
                        ])
                    ])
                ],color="#31343b"),],width=12)
            ],className="my-2"),
        ], width={'size': 2}, className="mb-5 mt-3"),
    ]),
], fluid=True)

@app.callback(
    Output('prod_forecast_callback', 'figure'),
    Input('pid_drop', 'value')
)
def graph_pred(pid):
    # Setting up the group by on the ProductName_ID
    grouped = pdf[pdf['ProductName_ID'].isin(pid)].groupby('week')['Units'].sum()
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

    prod_forecast = go.Figure()
    prod_forecast.add_trace(go.Scatter(x=lim_df.index, y=lim_df,
                             mode='lines',
                             name='TimeSeries'))
    prod_forecast.add_trace(go.Scatter(x=predictions.index, y=predictions,
                             mode='lines',
                             name='Predictions'))

    prod_forecast.update_layout(paper_bgcolor='rgba(255,255,255)', plot_bgcolor='rgba(0,0,0,0)')
    prod_forecast.update_xaxes(showline=True, linewidth=1, linecolor='black')
    prod_forecast.update_yaxes(showline=True, linewidth=1, linecolor='black')
    prod_forecast.update_xaxes(showgrid=True, gridwidth=0.1, gridcolor='lightskyblue')
    return prod_forecast

@app.callback(
    [
        Output('mape_prod', 'children'),
        Output('rmse_prod', 'children'),
    ],
    Input('pid_drop', 'value'),
)

def indicators(pid):
    # Setting up the group by on the ProductName_ID
    grouped = pdf[pdf['ProductName_ID'].isin(pid)].groupby('week')['Units'].sum()
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
        predictions = pd.Series(predictions, index=test_data.index)
    residuals = test_data - predictions

    MAPE = round(np.mean(abs(residuals/test_data)), 4)
    RMSE = round(np.sqrt(np.mean(residuals ** 2)), 4)

    return str(MAPE),\
           str(RMSE)