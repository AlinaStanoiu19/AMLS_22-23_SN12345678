from pre_processing_A import get_data_a, get_data_b
from A1.A1 import model_a1
from A2.A2 import model_a2
from B1.B1 import model_b1
from feature_extraction import face_detection
import numpy as np
import pandas as pd
import cv2
import pickle

def main():
    X_train_features = []
    X_test_features = []

    file_to_read = open("train_features_task_b1", "rb")
    train_features = pickle.load(file_to_read)
    file_to_read.close()
    for image in train_features.keys():
        X_train_features.append(train_features[image])
    X_train_features = np.array(X_train_features)
    file_to_read = open("test_features_task_b1", "rb")
    test_features = pickle.load(file_to_read)
    file_to_read.close()
    for image in test_features.keys():
        X_test_features.append(test_features[image])
    X_test_features = np.array(X_test_features)
    Y_test_labels, Y_train_labels = get_data_b("face_shape")
    
    model_b1(Y_train_labels, Y_test_labels, X_test_features, X_train_features, "face_shape_classification")

    # get data for task A1
    # Y_train_a1, Y_test_a1, X_test_a1, X_train_a1, train_images, test_images = get_data('gender')

    # feature extraction for task A1 and A2
    # X_train_features, X_train_landmarks, training_images_shape = face_detection(train_images, None)

    # X_test_features, X_test_landmarks, _ = face_detection(test_images,training_images_shape)

    # get data for task A2 
    # Y_train_a2, Y_test_a2, X_test_a2, X_train_a2, _, _  = get_data_a('smiling')
    # X_test_features, X_test_landmarks, _ = face_detection(test_images,training_images_shape)


    # print("TASK A1. GENDER PREDICTION MODEL WITH RAW DATA")
    # model_a(Y_train_a1, Y_test_a1, X_test_a1, X_train_a1, "gender")
    # print("TASK A1. GENDER PREDICTION MODEL WITH FEATURES EXTRACTION")
    # model_a(Y_train_a1, Y_test_a1, X_test_features, X_train_features, "gender")
    
    
    # print("TASK A2. SMILE PREDICTION MODEL WITH RAW DATA")
    # model_a2(Y_train_a2, Y_test_a2, X_test_a2, X_train_a2, "smiling")
    # print("TASK A2. SMILE PREDICTION MODEL WITH FEATURES EXTRACTION")
    # model_a(Y_train_a2, Y_test_a2, X_test_features, X_train_features, "smiling")


if __name__ == "__main__":
    main()