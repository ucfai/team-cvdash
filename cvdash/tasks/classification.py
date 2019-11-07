"""Functions for Classification tab
"""
from io import BytesIO

import cv2
import numpy as np
import plotly.graph_objects as go
import requests
from PIL import Image
from tensorflow.keras.applications import resnet_v2, vgg16, xception

from .. import utils

# module, model, input shape
models = {"xception": (xception, xception.Xception(), (299, 299)),
          "vgg16": (vgg16, vgg16.VGG16(), (224, 224)),
          "resnet50": (resnet_v2, resnet_v2.ResNet50V2(), (224, 224))}


def make_plotly_plot(pred, model_name):
    """
    Make the plotly plot given the predictions and model name.
    """
    _, y, x = zip(*pred[::-1])

    trace1 = go.Bar(x=x, y=y, orientation='h', text=x,
                    textfont=dict(size=24), textposition='auto')
    layout = go.Layout(title="Predictions from " + model_name, font=dict(size=24),autosize=False,
    width=800,
    height=500+(len(x)*25))
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

    # Get and decode predictions
    pred = model.predict(np.expand_dims(img, 0))
    pred = module.decode_predictions(pred, top=top)[0]

    fig = make_plotly_plot(pred, model_name)
    return fig


if __name__ == "__main__":
    img = utils.get_image("https://upload.wikimedia.org/wikipedia/commons/6/66/"
                    "An_up-close_picture_of_a_curious_male_domestic_shorthair_tabby_cat.jpg")
    fig = classification_plot(img, model_name="resnet50")
    fig.show()
    fig = classification_plot(img, model_name="vgg16")
    fig.show()
    fig = classification_plot(img, model_name="xception")
    fig.show()
