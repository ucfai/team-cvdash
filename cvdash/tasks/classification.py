"""Functions for Classification tab
"""
import cv2
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.applications import resnet_v2, vgg16, xception

# module, model, input shape
models = {
    "xception": (xception, xception.Xception(), (299, 299)),
    "vgg16": (vgg16, vgg16.VGG16(), (224, 224)),
    "resnet50": (resnet_v2, resnet_v2.ResNet50V2(), (224, 224)),
}


def make_plotly_plot(pred, model_name):
    """
    Make the plotly plot given the predictions and model name.
    """
    _, y, x = zip(*pred[::-1])
    x = [round(i * 100, 2) for i in x]

    trace1 = go.Bar(
        x=x,
        y=y,
        orientation="h",
        text=[str(i) + "%" for i in x],
        textfont=dict(size=20),
        textangle=0,
        textposition="auto",
    )

    layout = go.Layout(
        title="Predictions from " + model_name,
        font=dict(size=20),
        autosize=False,
        width=500,
        height=500 + (len(x) * 25),
    )
    fig = go.Figure(data=[trace1], layout=layout)

    return fig


def classification_plot(img, model_name, top=5):
    """
    Return a plotly plot of decided predictions

    Assume img is Pillow object
    """
    module, model, input_shape = models[model_name]

    # Preprocess image
    img = np.array(img)
    img = cv2.resize(img, input_shape)
    img = module.preprocess_input(img)
    img = img[:, :, :3]

    # Get and decode predictions
    pred = model.predict(np.expand_dims(img, 0))
    pred = module.decode_predictions(pred, top=top)[0]

    fig = make_plotly_plot(pred, model_name)

    return fig
