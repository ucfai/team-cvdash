# %% [markdown]
# # Pretrained Models
#
# Choose any model you want to use. Keep in mind some models use different preprocessing.
#
# See https://github.com/keras-team/keras-applications/tree/master/keras_applications
#
# ```python
# """Preprocesses a tensor encoding a batch of images.
# # Arguments
#     x: Input tensor, 3D or 4D.
#     data_format: Data format of the image tensor.
#     mode: One of "caffe", "tf" or "torch".
#         - caffe: will convert the images from RGB to BGR,
#             then will zero-center each color channel with
#             respect to the ImageNet dataset,
#             without scaling.
#         - tf: will scale pixels between -1 and 1,
#             sample-wise.
#         - torch: will scale pixels between 0 and 1 and then
#             will normalize each channel with respect to the
#             ImageNet dataset.
# # Returns
#     Preprocessed tensor.
# """
# ```

# %%
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from tensorflow.keras.applications.xception import (Xception,
                                                    decode_predictions,
                                                    preprocess_input)

# %%
# Input shape for model. Image should match model input size
input_shape = (299, 299)

def get_image(link):
    """
    Get image from link
    """
    r = requests.get(link)
    r.raise_for_status()
    return np.array(Image.open(BytesIO(r.content)))

img = get_image("https://upload.wikimedia.org/wikipedia/commons/6/66/"
                "An_up-close_picture_of_a_curious_male_domestic_shorthair_tabby_cat.jpg")

# %%
print(img.shape)
plt.imshow(img)

# %%
# Resize to target shape
img = cv2.resize(img, input_shape)
plt.imshow(img)

# %%
# Xception uses tf preprocessing
img = preprocess_input(img)

# %%
# Note that 299 x 299 is default shape for xception
model = Xception()

# Need a 4th dim for samples
pred = model.predict(np.expand_dims(img, 0))

# %%
decode_predictions(pred, top=5)

# %%
