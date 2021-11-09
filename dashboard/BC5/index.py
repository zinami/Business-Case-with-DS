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

import EDA
import MBA
import forecast
import forecast_pos
import forecast_prod
import cluster_pos
import cluster_value
import cluster_product

from app import app

nav_item_EDA = dbc.NavItem(dbc.NavLink("EDA", href="/EDA", active="exact"))
nav_item_MBA = dbc.NavItem(dbc.NavLink("MBA", href="/MBA", active="exact"))
nav_item_forecast = dbc.NavItem(dbc.NavLink("Forecast", href="/forecast", active="exact"))
nav_item_forecast_pos = dbc.NavItem(dbc.NavLink("Forecast(POS)", href="/forecast_pos", active="exact"))
nav_item_forecast_prod = dbc.NavItem(dbc.NavLink("Forecast(Prod)", href="/forecast_prod", active="exact"))
#nav_item_cluster_pos = dbc.NavItem(dbc.NavLink("Cluster (PoS)", href="/cluster_pos", active="exact"))
nav_item_cluster_value = dbc.NavItem(dbc.NavLink("Cluster Value", href="/cluster_value", active="exact"))
nav_item_cluster_product = dbc.NavItem(dbc.NavLink("Cluster Product", href="/cluster_product", active="exact"))


CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

logo = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        html.Img(src=app.get_asset_url("img.png"), height="70px",
                                 className="mr-auto"),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="https://www.formula1.com/",
            ),
            dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    [nav_item_EDA, nav_item_MBA, nav_item_forecast,nav_item_forecast_pos,
                     nav_item_forecast_prod,
                     #nav_item_cluster_pos,
                     nav_item_cluster_value,
                     nav_item_cluster_product], className="ml-auto", navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
        ],fluid=True
    ),
    color="#31343b",
    dark=True,
    className="mb-3",
)

content = html.Div(id="page-content")

app.layout = html.Div(
    [dcc.Location(id="url"), logo, content],
)


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
       return EDA.layout
    elif pathname == "/EDA":
        return EDA.layout
    elif pathname == "/MBA":
        return MBA.layout
    elif pathname == "/forecast":
        return forecast.layout
    elif pathname == "/forecast_pos":
        return forecast_pos.layout
    elif pathname == "/forecast_prod":
        return forecast_prod.layout
    #elif pathname == "/cluster_pos":
    #    return cluster_pos.layout
    elif pathname == "/cluster_value":
        return cluster_value.layout
    elif pathname == "/cluster_product":
        return cluster_product.layout
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

if __name__ == '__main__':
    app.run_server(debug=True)