import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import json
import pandas as pd
import numpy as np
import plotly
import mysql.connector


app = dash.Dash()

blackjack_df = pd.read_pickle('/Users/befeltingu/DeepRL/Data/DataFrames/blackjack_policy0')

app.layout = html.Div([

    html.Div([

        html.Div([

        ],className="col-sm-4"),

        html.Div([

        ], className="col-sm-8")

    ],className="row"),

    html.H4('Gapminder DataTable'),

    dt.DataTable(
        rows=blackjack_df.to_dict('records'),
        # optional - sets the order of columns
        columns=blackjack_df.columns,
        row_selectable=True,
        filterable=True,
        sortable=True,
        selected_row_indices=[],
        id='datatable-gapminder'
    ),

    html.Div(id='selected-indexes'),

], className="container")

'''
@app.callback(
    Output('datatable-gapminder', 'selected_row_indices'),
    [Input('graph-gapminder', 'clickData')],
    [State('datatable-gapminder', 'selected_row_indices')])
def update_selected_row_indices(clickData, selected_row_indices):
    if clickData:
        for point in clickData['points']:
            if point['pointNumber'] in selected_row_indices:
                selected_row_indices.remove(point['pointNumber'])
            else:
                selected_row_indices.append(point['pointNumber'])
    return selected_row_indices
'''

if __name__ == '__main__':


    server = app.server

    app.config.suppress_callback_exceptions = True

    app.run_server(debug=True)