# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.applications.xception import Xception, decode_predictions

# %%
input_shape = (299, 299, 3)

# %%
img = cv2.imread("cat.jpg")[:, :, ::-1]

# %%
plt.imshow(img)

# %%
img = cv2.resize(img, input_shape[:-1])

# %%
img = img / 255

# %%
model = Xception()

# %%
pred = model.predict(np.expand_dims(img, 0))

# %%
decode_predictions(pred)

# %%
