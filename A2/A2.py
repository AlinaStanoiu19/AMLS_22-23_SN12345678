import cv2
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout,MaxPooling2D
from sklearn.model_selection import train_test_split



def model_a2(Y_train, Y_test, X_test, X_train, model_name):
    print(Y_train[:10])
    input_shape = X_train[0].shape
    print(f"input shape is {input_shape}")
    model = Sequential()
    model.add(Conv2D(input_shape=input_shape, activation='relu',
            filters=32, kernel_size=3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64,activation='relu', kernel_size=3, ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy",
                metrics=["accuracy"])

    epochs = 12
    batch_size = 16
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train,Y_train, test_size=0.2)
    print(f'the validation set has {len(Y_valid)} length')

    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_valid, Y_valid))

    model.save(model_name)

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(X_test, Y_test, batch_size=10)
    print("test loss, test acc:", results)


    y_predictions = np.array(model.predict(X_test, verbose=1, batch_size=10))
    print(y_predictions[0])

    y_predictions_binary = np.where(y_predictions > 0.5, 1, 0)

    df_predictions = pd.DataFrame(y_predictions)
    df_predictions_binary = pd.DataFrame(y_predictions_binary)
    df_predictions.to_csv(model_name + "prediction_float.csv", index=False, header=None)
    df_predictions_binary.to_csv(model_name + "prediction_binary.csv", index=False, header=None)

    correct_predict = 0

    for i in range(len(y_predictions_binary)):
        if y_predictions_binary[i] == Y_test[i]:
            correct_predict += 1

    print(f"accuracy of the prediction for {model_name} is: {correct_predict/len(y_predictions_binary)}")