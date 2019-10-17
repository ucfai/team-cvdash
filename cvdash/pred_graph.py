from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests

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



def plot_preds(img):
    img = cv2.resize(img, input_shape)
    img = preprocess_input(img)
    model = Xception()
    pred = model.predict(np.expand_dims(img, 0))

    return decode_predictions(pred, top=5)



if __name__ == "__main__":
    img = get_image("https://upload.wikimedia.org/wikipedia/commons/6/66/"
                    "An_up-close_picture_of_a_curious_male_domestic_shorthair_tabby_cat.jpg")
    print(plot_preds(img))
