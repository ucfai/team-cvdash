#!/usr/bin/env python
import datetime
import os

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

assets_path = os.getcwd() + "/cvdash/assets"
#print(assets_path)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, 
    #assets_folder=assets_path
    )

init_k_val = 5
max_k_val = 20

temp_image = int


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
    html.Div(id='container',
        children=[
            dcc.Upload(
            id='upload-image',
            children=[html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ])]
            ,

            style={
                'align': 'left',
                'width': '45%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                #'margin': '10px',
                'margin': '1em',
                'float': 'left'
            },
            # Allow multiple files to be uploaded
            multiple=True
            ),
            html.Div(
            id='slider-div',
            children=[
                dcc.Slider(
                id='k-slider',
                min=3,
                max=max_k_val,
                step=1,
                value=init_k_val,
                marks={str(i) : str(i) for i in range(3, max_k_val+1)}
                )
            ],
            style={
                'width': '45%',
                'float': 'right',
                'align': 'right',
                'margin': '1em'
            }
            )],
        style={
           'clear': 'both'
        }
        )
    ,
    html.Div(id='container2', children=[
        html.Div(
            id='output-image-upload',
            style={
                'width': '45%',
                'display': 'inline-block',
                'float': 'left',
                'margin': '1em'
            },
            children=[parse_contents(app.get_asset_url('cat.jpg'))]
        ),
        html.Div([dcc.Graph(
            id='bar_graph',
            figure=classification.classification_plot(
                utils.get_image(utils.example_image_link), 'xception') 
        )],
        style={
            'width': '45%',
            'display': 'inline-block',
            'float': 'right',
            'margin': '1em'
        }
        )
        ],
        style={
            'clear': 'both'
        })

])


@app.callback([Output('output-image-upload', 'children'),
            Output('bar_graph','figure')],
              [Input('upload-image', 'contents'), 
              Input('k-slider', 'value')],
              [State('output-image-upload', 'children'),
              State('bar_graph', 'figure'),
              #State('k-slider', 'value')
              ]
              )
def update_output(uploaded_image, k_val, state_img, state_bg):
    #print(list_of_contents)
    #print(len(list_of_contents))
    '''
    if list_of_contents is not None:
        
        # no image upload
        if(list_of_contents[0] == None):
            if()
            temp_image = list_of_contents[2]
            print(temp_image)
            k = list_of_contents[1]
            plot = classification.classification_plot(
                temp_image, 'xception', top=k)
            return [list_of_contents[2],plot]
        # has image upload
        else:
            plot = classification.classification_plot(
                utils.b64_to_PIL(contents[23:]), 'xception')
            contents = list_of_contents[0][0]
            children = parse_contents(contents)
            return [children, plot]
    else:
        raise PreventUpdate
    '''
    print(uploaded_image, k_val)
    if(uploaded_image is None and k_val ==init_k_val):
        raise PreventUpdate

    if(uploaded_image is not None):
        plot = classification.classification_plot(
                utils.b64_to_PIL(uploaded_image[0][23:]), 'xception')
        children = parse_contents(uploaded_image[0][23:])
        return [children, plot]
    if(k_val != init_k_val):
        #plot = classification.classification_plot(
        #        utils.b64_to_PIL(uploaded_image[0][23:]), 'xception')
        if(type(state_img[0]) == str):
            plot = classification.classification_plot(
                utils.b64_to_PIL(state_img[0][23:]), 'xception', top=k_val)
            return [state_img, plot]

        if(type(state_img[0]) == dict):
            plot = classification.classification_plot(
                utils.get_image(utils.example_image_link), 'xception', top=k_val) 
            return [state_img, plot]




if __name__ == '__main__':
    app.run_server(debug=True)
    #app.get_asset_url("cvdash/assets")
