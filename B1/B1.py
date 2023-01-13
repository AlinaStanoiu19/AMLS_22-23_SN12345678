import cv2
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def model_b1(Y_train, Y_test, X_test, X_train, model_name):
    h, w, _ = X_train[0].shape
    model = Sequential()
    model.add(Conv2D(input_shape=(85,224,3,),activation='relu', filters=32, kernel_size=3, strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64,activation='relu', kernel_size=3, strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128,activation='relu', kernel_size=3, strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy",
                metrics=["accuracy"])

    epochs = 12
    batch_size = 15
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train,Y_train, test_size=0.2)
    # X_train, X_valid, Y_train, Y_valid = train_test_split(X_train,Y_train, test_size=0.2, random_state=42)
    print(f'the validation set has {len(Y_valid)} length')
    print(f'the train set has {len(X_train)} length')

    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_valid, Y_valid))

    model.save(model_name)

    # load model
    model = load_model('b2_feature_data')
    # summarize model.
    model.summary()

    X_test_resized = []
    for i in range(len(X_test)):
        X_test_resized.append(cv2.resize(X_test[i],(224,85)))

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(np.array(X_test_resized), Y_test, batch_size=None)
    print("test loss, test acc:", results)