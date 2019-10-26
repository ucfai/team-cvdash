#!/usr/bin/env python
import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from cvdash.tasks import classification
from cvdash import utils

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

colors = {
    'background': '343434'
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def parse_contents(contents):
    return html.Div([
        html.Img(src=contents,
            style={
                'max-width': '100%',
                'max-height': '100%',
                'align': 'middle'
            }),
    ])


app.layout = html.Div(
    style={'backgroundColor': colors['background']},
    children=[
    html.Div(id='title', children=html.H1(children='CVDash'),
        style={
            'textAlign': 'center'
        }),
    dcc.Upload(
        id='upload-image',
        children=[html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ])],

        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(
        id='output-image-upload',
        style={
            'width': '45%',
            'display': 'inline-block',
            'margin': '1em'
        },
        children=[parse_contents(app.get_asset_url('cat.jpg'))]
    ),
    html.Div([dcc.Graph(
            id='bar_graph',
            figure=classification.classification_plot(
                utils.get_image(utils.example_image_link), 'xception') 
        )
    ],
        style={
            'width': '45%',
            'display': 'inline-block',
            'float': 'right',
            'margin': '1em'
        }
    )
])


@app.callback([Output('output-image-upload', 'children'),
            Output('bar_graph','figure')],
              [Input('upload-image', 'contents')],
              [State('output-image-upload', 'children'),
              State('bar_graph', 'figure')]
              )
def update_output(*list_of_contents):
    if None not in list_of_contents:
        contents = list_of_contents[0][0]
        children = parse_contents(contents)

        plot = classification.classification_plot(
            utils.b64_to_PIL(contents[23:]), 'xception')
        
        return [children, plot]
    else:
        raise PreventUpdate


if __name__ == '__main__':
    app.run_server(debug=True)
    app.get_asset_url("cvdash/assets/")
