import datetime
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import model

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div(id='title',children=html.H1(children='CVDash'),
        style={
            'textAlign':'center'
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
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),  
    html.Div(
        id='output-image-upload',
        style={
            'width': '40%',
            'display': 'inline-block',
            'margin': '1em'
        }
    ),
    html.Div([dcc.Graph(
            id='example-graph',
            figure={
                'data': [
                    {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                    {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
                ],
                'layout': {
                    'title': 'Dash Data Visualization'
                }
            })
        ],
        style={
            'width':'40%',
            'display':'inline-block',
            'float': 'right',
            'margin': '1em'
        }
    )
])


def parse_contents(contents):
    return html.Div([
        #html.H5(filename),
        #html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        #html.Hr(),
        #html.Div('Raw Content'),
        #html.Pre(contents[0:200] + '...', style={
        #    'whiteSpace': 'pre-wrap',
        #    'wordBreak': 'break-all'
        #})
    ])


@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        #children = [
        #    parse_contents(c, n, d) for c, n, d in
        #    zip(list_of_contents, list_of_names, list_of_dates)]
        children = [parse_contents(item) for item in list_of_contents]


        string_b64 = str(list_of_contents[0])[23:]
        image = b64_to_np(string_b64)
        model.main(image, 5)

        return children

def b64_to_np(string):
    x = base64.b64decode(string)
    decoded = BytesIO(x)

    image = Image.open(decoded)

    im = np.array(image, dtype=np.float32)

    return im


if __name__ == '__main__':
    app.run_server(debug=True)