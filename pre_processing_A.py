import cv2
import csv
import os
import numpy as np
import pandas as pd
from feature_extraction import cartoon_face_detection, face_shape_feature_extraction, eye_feature_extraction
import pickle

DIR_TRAIN = "Datasets\dataset_AMLS_22-23\celeba"
DIR_TEST = "Datasets\dataset_AMLS_22-23_test\celeba_test"
DIR_TRAIN_B = "Datasets\dataset_AMLS_22-23\cartoon_set"
DIR_TEST_B = "Datasets\dataset_AMLS_22-23_test\cartoon_set_test"

# Get data for task A1 and A2
# output: dictionaries with image name as key and  image/label as values
# output: np araays for images and labels in order 

def read_labels(dir, column):
    df = pd.read_csv(dir + "\labels.csv", sep="\t",index_col=0)
    Y_labels = []
    Y_labels_dict = {}
    for i in range(len(df)):
        if df[column][i] == 1:
            Y_labels.append(1)
            Y_labels_dict[df["img_name"][i]] = 1
        else:
            Y_labels.append(0)
            Y_labels_dict[df["img_name"][i]] = 0

    return Y_labels, Y_labels_dict

def read_labels_cartoons(dir, column):
    df = pd.read_csv(dir + "/labels.csv", sep="\t",index_col=0)
    Y_labels = []
    Y_labels_dict = {}
    for i in range(len(df)):
        if df[column][i] == 0:
            Y_labels.append([1, 0, 0, 0, 0])
            Y_labels_dict[df["file_name"][i]] = [1, 0, 0, 0, 0]
        elif df[column][i] == 1:
            Y_labels.append([0, 1, 0, 0, 0])
            Y_labels_dict[df["file_name"][i]] = [0, 1, 0, 0, 0]
        elif df[column][i] == 2:
            Y_labels.append([0, 0, 1, 0, 0])
            Y_labels_dict[df["file_name"][i]] = [0, 0, 1, 0, 0]
        elif df[column][i] == 3:
            Y_labels.append([0, 0, 0, 1, 0])
            Y_labels_dict[df["file_name"][i]] = [0, 0, 0, 1, 0]
        elif df[column][i] == 4:
            Y_labels.append([0, 0, 0, 0, 1])
            Y_labels_dict[df["file_name"][i]] = [0, 0, 0, 0, 1]

    return Y_labels, Y_labels_dict

def read_images(dir, gray_rgb):

    image_names = []
    file_type = '.' + os.listdir(dir)[0].split(".")[1]
    print(file_type)
    for image in os.listdir(dir):
        image_names.append(image.split(".")[0])
    image_names.sort(key=int)
    sorted_image_names = [x + file_type for x in image_names]

    X_images = []
    X_images_dict = {}
    for image in sorted_image_names:
        if gray_rgb == 'rgb':
            img = cv2.imread(dir + image)
        elif gray_rgb == 'gray':
            img = cv2.imread(dir + image, cv2.IMREAD_GRAYSCALE)
        X_images.append(img)
        X_images_dict[image] = img
    
    return X_images, X_images_dict


def get_data_a(task):


    # get training and testing labels with one-hot_encoding 
    Y_train_labels, _ = read_labels(DIR_TRAIN, task)
    Y_test_labels,  _= read_labels(DIR_TEST, task)

    # Get training and testing images 
    X_train_images, train_images = read_images(DIR_TRAIN + '\img\\','rgb')
    X_test_images, test_images = read_images(DIR_TEST + '\img\\','rgb')


    Y_train_labels = np.array(Y_train_labels)
    Y_test_labels = np.array(Y_test_labels)
    X_test_images = np.array(X_test_images)
    X_train_images = np.array(X_train_images)

    return Y_train_labels, Y_test_labels, X_test_images, X_train_images, train_images, test_images


def get_data_b(task):

    # FEATURE EXTRACTION FOR TASK B2 ------------------------------------------------

    # get training and testing labels 
    Y_train_labels, training_labels = read_labels_cartoons(DIR_TRAIN_B, task)
    Y_test_labels,  testing_labels= read_labels_cartoons(DIR_TEST_B, task)
    # convert training labels in numpy
    Y_train_labels = np.array(Y_train_labels)
    Y_test_labels = np.array(Y_test_labels)
    print("labels have been loaded")
    # Get training and testing images - raw data
    X_train_images, train_images = read_images(DIR_TRAIN_B + '\img\\','rgb')
    X_test_images, test_images = read_images(DIR_TEST_B + '\img\\','rgb')
    # convert training data into numpy array 
    X_train_images = np.array(X_train_images)
    X_test_images = np.array(X_test_images)
    print("images have been loaded ")
    # feature extraction for training set ------------------------
    print("Strat feature extraction for B2 for training data")
    images, gray_images, landmarks_dict, no_faces_list = cartoon_face_detection(train_images)
    training_features, sunglasses_images = eye_feature_extraction(images, gray_images, landmarks_dict, no_faces_list)
    # remove outliers with glasses from raw data, feature data and labels
    Y_train_removed_glasses = []
    X_train_removed_glasses = []
    for image in training_labels.keys():
        if image not in sunglasses_images:
            Y_train_removed_glasses.append(training_labels[image])
            X_train_removed_glasses.append(train_images[image])
    X_train_features = []
    for image in training_features.keys():
        if image not in sunglasses_images:
            X_train_features.append(training_features[image])

    # get the training data with glasses removed and feature exctraction as numpy array 
    Y_train_removed_glasses = np.array(Y_train_removed_glasses)
    X_train_removed_glasses = np.array(X_train_removed_glasses)
    X_train_features = np.array(X_train_features)

    print("write the training features without glasses to file")
    outfile_features = open("x_train_features_removed_glasses_task_b2", 'wb')
    pickle.dump(X_train_features, outfile_features)
    outfile_features.close()

    print("write the training labels withou glasses to file")
    outfile_features = open("x_train_removed_glasses_task_b2", 'wb')
    pickle.dump(X_train_removed_glasses, outfile_features)
    outfile_features.close()

    print("write the training labels withou glasses to file")
    outfile_features = open("y_train_removed_glasses_task_b2", 'wb')
    pickle.dump(Y_train_removed_glasses, outfile_features)
    outfile_features.close()

    print("Strat feature extraction for B2 for testing data")
    images, gray_images, landmarks_dict, no_faces_list = cartoon_face_detection(test_images)
    testing_features, sunglasses_images = eye_feature_extraction(images, gray_images, landmarks_dict, no_faces_list)
    # remove outliers with glasses from raw data, feature data and labels
    Y_test_removed_glasses = []
    X_test_removed_glasses = []
    for image in testing_labels.keys():
        if image not in sunglasses_images:
            Y_test_removed_glasses.append(testing_labels[image])
            X_test_removed_glasses.append(test_images[image])
    X_test_features = []
    for image in testing_features.keys():
        if image not in sunglasses_images:
            X_test_features.append(testing_features[image])

    # get the training data with glasses removed and feature exctraction as numpy array 
    Y_test_removed_glasses = np.array(Y_test_removed_glasses)
    X_test_removed_glasses = np.array(X_test_removed_glasses)
    X_test_features = np.array(X_test_features)

    print("write the testing features without glasses to file")
    outfile_features = open("x_test_features_removed_glasses_task_b2", 'wb')
    pickle.dump(X_test_features, outfile_features)
    outfile_features.close()

    print("write the testing labels withou glasses to file")
    outfile_features = open("x_test_removed_glasses_task_b2", 'wb')
    pickle.dump(X_test_removed_glasses, outfile_features)
    outfile_features.close()

    print("write the testing labels withou glasses to file")
    outfile_features = open("y_test_removed_glasses_task_b2", 'wb')
    pickle.dump(Y_test_removed_glasses, outfile_features)
    outfile_features.close()

    # FEATURE EXTRACTION FOR TASK B1 ------------------------------------------------

    # print("start feature extraction for training set")
    # images, gray_images, landmarks_dict, no_faces_list = cartoon_face_detection(train_images)
    # features = face_shape_feature_extraction(gray_images, landmarks_dict, no_faces_list)
    
    # print("write the features to file")
    # outfile_features = open("train_features_task_b1", 'wb')
    # pickle.dump(features, outfile_features)
    # outfile_features.close()

    # outfile_images = open("train_images_rgb", 'wb')
    # pickle.dump(images, outfile_images)
    # outfile_images.close()

    # outfile_images = open("train_images_gray", 'wb')
    # pickle.dump(images, outfile_images)
    # outfile_images.close()

    # print("start feature extraction for testing set")
    # images, gray_images, landmarks_dict, no_faces_list = cartoon_face_detection(test_images)
    # features = face_shape_feature_extraction(gray_images, landmarks_dict, no_faces_list)
    # print("write the features to file")
    # outfile_features = open("test_features_task_b1", 'wb')
    # pickle.dump(features, outfile_features)
    # outfile_features.close()

    # outfile_images = open("test_images_rgb", 'wb')
    # pickle.dump(images, outfile_images)
    # outfile_images.close()

    # outfile_images = open("test_images_gray", 'wb')
    # pickle.dump(images, outfile_images)
    # outfile_images.close()

    return Y_test_labels, Y_train_labels


