from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import plotly.offline as py
import plotly.graph_objects as go

from PIL import Image
from tensorflow.keras.applications.xception import (Xception,
                                                    decode_predictions,
                                                    preprocess_input)

input_shape = (299, 299)

def get_image(link):
    """
    Get image from link
    """
    r = requests.get(link)
    r.raise_for_status()
    return np.array(Image.open(BytesIO(r.content)))



def make_plotly_plot(pred):
    _, y, x = zip(*pred[::-1])

    trace1 = go.Bar(x=x, y=y, orientation='h', text=x,
                    textfont=dict(size=20), textposition='auto')
    layout = go.Layout(title="Predictions", font=dict(size=24))
    fig = go.Figure(data=[trace1], layout=layout)
    return fig

def classification_plot(img, model="xception", top=5):
    """
    Return a plotly plot of decided predictions

    Assume img is Pillow object
    """
    img = np.array(img)
    img = cv2.resize(img, input_shape)

    img = preprocess_input(img)
    model = Xception()
    pred = model.predict(np.expand_dims(img, 0))
    pred = decode_predictions(pred, top=top)[0]

    fig =  make_plotly_plot(pred)
    return fig



if __name__ == "__main__":
    img = get_image("https://upload.wikimedia.org/wikipedia/commons/6/66/"
                    "An_up-close_picture_of_a_curious_male_domestic_shorthair_tabby_cat.jpg")
    fig = classification_plot(img, top=10)
    fig.show()