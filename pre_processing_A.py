import cv2
import csv
import os
import numpy as np
import pandas as pd

# Get data for task A1 and A2

def labels_to_df(dir):
    # empty dataframe to store the labelsfor both tasks using one-hot-encoding
    df_labels = pd.DataFrame(columns=['img_name', 'gender', 'smiling'])
    # read the labels file 
    with open(dir + "\labels.csv", "r", encoding="utf8") as labels_file: 
        # set the separator to tab
        tsv_reader = csv.reader(labels_file, delimiter="\t")

        # skip first row with headers 
        next(tsv_reader)

        # read each row from the file
        for row in tsv_reader:
            df_labels.loc[row[0]] = [row[1], row[2], row[3]]

    return df_labels

def labels_encoding(df, column):
    Y_labels = []
    for i in range(len(df)):
        if df[column][i] == '1':
            Y_labels.append([1, 0])
        else:
            Y_labels.append([0, 1])

    return Y_labels

def read_images(dir):

    image_names = []
    for image in os.listdir(dir):
        image_names.append(image.split(".")[0])
    image_names.sort(key=int)
    sorted_image_names = [x + '.jpg' for x in image_names]

    X_images = []
    for image in sorted_image_names:
        img = cv2.imread(dir + image)
        X_images.append(img)
    
    return X_images

def get_data(task):
    dir_train = "Datasets\dataset_AMLS_22-23\celeba"
    dir_test = "Datasets\dataset_AMLS_22-23_test\celeba_test"

    # Get the labels in a dataframe
    df_labels_train = labels_to_df(dir_train)
    df_labels_test = labels_to_df(dir_test)

    # get training labels with one-hot_encoding for task A1
    Y_train_labels = labels_encoding(df_labels_train, task)

    # get testing labels with one-hot-encoding for task A2 
    Y_test_labels = labels_encoding(df_labels_test, task)

    print(Y_test_labels[2:10])

    # Get the images
    X_train_images = read_images(dir_train + '\img\\')
    X_test_images = read_images(dir_test + '\img\\')

    Y_train_labels = np.array(Y_train_labels)
    Y_test_labels = np.array(Y_test_labels)
    X_test_images = np.array(X_test_images)
    X_train_images = np.array(X_train_images)

    return Y_train_labels, Y_test_labels, X_test_images, X_train_images