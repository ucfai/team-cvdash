#!/usr/bin/env python
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from cvdash import utils
from cvdash.tasks import classification

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
colors = {"background": "343434"}
assets_path = os.getcwd() + "/cvdash/assets"

app = dash.Dash(
    __name__, external_stylesheets=external_stylesheets, assets_folder=assets_path
)

INIT_TOP_K = 5
MIN_TOP_K = 3
MAX_TOP_K = 20


def parse_contents(contents):
    return html.Div(
        [
            html.Img(
                src=contents,
                style={"max-width": "100%", "max-height": "100%", "align": "middle"},
            )
        ]
    )


app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.Div(
            id="title",
            children=html.H1(children="CVDash"),
            style={"textAlign": "center"},
        ),
        html.Div(
            id="major_container1",
            children=[
                dcc.Upload(
                    id="upload-image",
                    children=[html.Div(["Drag and Drop or ", html.A("Select Files")])],
                    style={
                        "align": "left",
                        "width": "45%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "1em",
                        "float": "left",
                    },
                    multiple=True,
                ),
                html.Div(
                    id="slider-div",
                    children=[
                        dcc.Slider(
                            id="k-slider",
                            min=MIN_TOP_K,
                            max=MAX_TOP_K,
                            step=1,
                            value=INIT_TOP_K,
                            marks={str(i): str(i) for i in range(3, MAX_TOP_K + 1)},
                        )
                    ],
                    style={
                        "width": "45%",
                        "float": "right",
                        "align": "right",
                        "margin": "1em",
                    },
                ),
            ],
            style={"clear": "both"},
        ),
        html.Div(
            id="major_container2",
            children=[
                html.Div(
                    id="output-image-upload",
                    style={
                        "width": "45%",
                        "display": "inline-block",
                        "float": "left",
                        "margin": "1em",
                    },
                    children=[parse_contents(app.get_asset_url("cat.jpg"))],
                ),
                html.Div(
                    id="right_side_container",
                    children=[
                        dcc.Dropdown(
                            id="model-dropdown",
                            options=[
                                {"label": "Xception", "value": "xception"},
                                {"label": "VGG 16", "value": "vgg16"},
                                {"label": "ResNet 50", "value": "resnet50"},
                            ],
                            value="xception",
                            clearable=False,
                            searchable=False
                        ),
                        dcc.Graph(
                            id="bar_graph",
                            figure=classification.classification_plot(
                                utils.get_image(utils.example_image_link), "xception"
                            ),
                        ),
                    ],
                    style={
                        "width": "45%",
                        "display": "inline-block",
                        "float": "right",
                        "margin": "1em",
                    },
                ),
            ],
            style={"clear": "both"},
        ),
    ],
)


@app.callback(
    [Output("output-image-upload", "children"), Output("bar_graph", "figure")],
    [
        Input("upload-image", "contents"),
        Input("k-slider", "value"),
        Input("model-dropdown", "value"),
    ],
    [State("output-image-upload", "children"), State("bar_graph", "figure")],
)
def update_output(uploaded_image, k_val, dropdn_val, state_img, state_graph):

    if uploaded_image is not None:
        plot = classification.classification_plot(
            utils.b64_to_PIL(uploaded_image[0][23:]), dropdn_val, top=k_val
        )
        children = parse_contents(uploaded_image[0])
        return [children, plot]

    if uploaded_image is None:
        plot = classification.classification_plot(
            utils.get_image(utils.example_image_link), dropdn_val, top=k_val
        )
        return [state_img, plot]


if __name__ == "__main__":
    app.run_server(debug=True)
