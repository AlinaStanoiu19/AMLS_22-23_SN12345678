# A1: Gender detection: male or female
# Celeba Dataset - 5000 images of CelebA set.
# the labels for the celeba dataset. The first column is the index. 
# The second column is the corresponding file name. The third column is the gender ({-1, +1}). 

import cv2
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten


def model_a1(Y_train, Y_test, X_test, X_train):

    model = Sequential()
    model.add(Conv2D(input_shape=(218, 178, 3),
            filters=32, kernel_size=3, strides=(1, 1)))
    model.add(Conv2D(filters=16, kernel_size=3, strides=(1, 1)))
    model.add(Conv2D(filters=8, kernel_size=3, strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer="adam", loss="binary_crossentropy",
                metrics=["accuracy"])

    epochs = 5
    batch_size = 64

    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    model.save("a1_model")

    y_predictions = np.array(model.predict(X_test))
    print(y_predictions[0])

    y_predictions_binary = np.where(y_predictions > 0.5, 1, 0)

    df_predictions = pd.DataFrame(y_predictions)
    df_predictions_binary = pd.DataFrame(y_predictions_binary)
    df_predictions.to_csv("gender_prediction_float.csv", index=False, header=None)
    df_predictions_binary.to_csv("gender_prediction_binary.csv", index=False, header=None)

    correct_predict = 0

    for i in range(len(y_predictions_binary)):
        if np.array(y_predictions_binary[i],Y_test[i] ):
            correct_predict += 1

    print(f"accuracy of the prediction is: {correct_predict/len(y_predictions_binary)}")

