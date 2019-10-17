from tensorflow.keras.applications.xception import Xception, decode_predictions
import numpy as np
import cv2


def main(image,k):
    input_shape = (299,299,3)

    model = Xception()

    img = cv2.resize(image, input_shape[:-1])
    img /= 255

    pred = model.predict(np.expand_dims(img,0))

    predictions = decode_predictions(pred, k)[0]
    
    for i in range(len(predictions)):
        predictions[i] = predictions[i][1:]
    print(predictions)
    return predictions
