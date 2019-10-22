import base64
from PIL import Image
from io import BytesIO
import requests
import numpy as np

example_image_link = "https://upload.wikimedia.org/wikipedia/commons/6/66/" \
            "An_up-close_picture_of_a_curious_male_domestic_shorthair_tabby_cat.jpg"


def get_image(link):
    """
    Get image from link
    """
    r = requests.get(link)
    r.raise_for_status()
    return np.array(Image.open(BytesIO(r.content)))


def np_to_b64(arr, altchars=None):
    if(altchars is not None):
        return altchars + base64.b64encode(arr).decode('ascii')
    return base64.b64encode(arr).decode('ascii')


def b64_to_np(string):
    x = base64.b64decode(string)
    decoded = BytesIO(x)

    image = Image.open(decoded)

    im = np.array(image, dtype=np.float32)

    return im


def b64_to_PIL(string):
    x = base64.b64decode(string)
    decoded = BytesIO(x)
    image = Image.open(decoded)
    return image