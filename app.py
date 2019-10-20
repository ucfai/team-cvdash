#!/usr/bin/env python
import base64
import datetime
from io import BytesIO

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output, State
from PIL import Image

from cvdash.tasks import classification

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

colors = {
    'background': '343434'
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

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
        }
    ),
    html.Div([dcc.Graph(
            id='bar_graph',
            figure=# inital graph
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


def parse_contents(contents):
    return html.Div([
        html.Img(src=contents,
            style={
                'max-width': '100%',
                'max-height': '100%',
                'align': 'middle'
            }),
    ])


@app.callback([Output('output-image-upload', 'children'),
            Output()],
              [Input('upload-image', 'contents')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        # children = [
        #    parse_contents(c, n, d) for c, n, d in
        #    zip(list_of_contents, list_of_names, list_of_dates)]
        children = [parse_contents(item) for item in list_of_contents]

        string_b64 = str(list_of_contents[0])[23:]
        image = b64_to_np(string_b64)
        classification.main(image, 5)

        return children


def b64_to_np(string):
    x = base64.b64decode(string)
    decoded = BytesIO(x)

    image = Image.open(decoded)

    im = np.array(image, dtype=np.float32)

    return im


def np_to_b64(arr):
    return base64.b64encode(arr)


def b64_to_PIL(string):
    x = base64.b64decode(string)
    decoded = BytesIO(x)
    image = Image.open(decoded)
    return image


if __name__ == '__main__':
    app.run_server(debug=True)
