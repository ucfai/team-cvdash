#!/usr/bin/env python
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from cvdash import utils
from cvdash.tasks import classification

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

colors = {"background": "343434"}

assets_path = os.getcwd() + "/cvdash/assets"

app = dash.Dash(
    __name__, external_stylesheets=external_stylesheets, assets_folder=assets_path
)

init_k_val = 5
max_k_val = 20


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
            id="container",
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
                            min=3,
                            max=max_k_val,
                            step=1,
                            value=init_k_val,
                            marks={str(i): str(i) for i in range(3, max_k_val + 1)},
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
            id="container2",
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
                    [
                        dcc.Graph(
                            id="bar_graph",
                            figure=classification.classification_plot(
                                utils.get_image(utils.example_image_link), "xception"
                            ),
                        )
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
    [Input("upload-image", "contents"), Input("k-slider", "value")],
    [State("output-image-upload", "children"), State("bar_graph", "figure")],
)
def update_output(uploaded_image, k_val, state_img, state_bg):
    if uploaded_image is None and k_val == init_k_val:
        raise PreventUpdate

    if uploaded_image is not None:
        plot = classification.classification_plot(
            utils.b64_to_PIL(uploaded_image[0][23:]), "xception", top=k_val
        )
        children = parse_contents(uploaded_image[0])
        return [children, plot]

    if uploaded_image is None and k_val != init_k_val:
        plot = classification.classification_plot(
            utils.get_image(utils.example_image_link), "xception", top=k_val
        )
        return [state_img, plot]


if __name__ == "__main__":
    app.run_server(debug=True)
