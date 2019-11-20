import base64
from io import BytesIO

import numpy as np
import requests
from PIL import Image

import base64
import requests

example_image_link = (
    "https://upload.wikimedia.org/wikipedia/commons/6/66/"
    "An_up-close_picture_of_a_curious_male_domestic_shorthair_tabby_cat.jpg"
)


def get_image(link):
    """
    Get image from link
    """
    r = requests.get(link)
    r.raise_for_status()
    bytes = BytesIO(r.content)
    nparray = Image.open(bytes)
    return np.array(nparray)

def np_to_PIL(arr):
    return Image.fromarray(arr)



def np_to_b64(arr, altchars=None):
    if altchars:
        return altchars + base64.b64encode(arr).decode("ascii")
    return base64.b64encode(arr).decode("ascii")


def b64_to_PIL(string):
    x = base64.b64decode(string)
    decoded = BytesIO(x)
    image = Image.open(decoded)
    return image

def add_image_header(string):
    position = string[::-1].find('.',0,len(string))
    return "data:image/" + string[-position:] + ";base64,"

def add_image_header2(url):
    response = requests.get(url)
    uri = ("data:" + response.headers['Content-Type'] + ";" + "base64," + base64.b64encode(response.content).decode("utf-8"))
    return uri
