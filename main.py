from pre_processing_A import get_data_a, get_data_b, save_as_pickle
from A1.A1 import model_a1
from A2.A2 import model_a2
from B1.B1 import model_b1
from feature_extraction import face_detection
import numpy as np
import pandas as pd
import cv2
import pickle


def main():

    # Compute and save feature extraction as pkl file  for task B 

    # TASK B2: -----------------------------------------------------------------------------------------
    Y_train_labels, X_train_images, X_train_features, Y_test_labels, X_test_images, X_test_features = get_data_b('eye_color')

    save_as_pickle(Y_train_labels, 'y_train_labels_b2.pkl' )
    save_as_pickle(X_train_images, 'x_train_images_b2.pkl' )
    save_as_pickle(X_train_features, 'x_train_features_b2.pkl' )
    save_as_pickle(Y_test_labels, 'y_test_labels_b2.pkl' )
    save_as_pickle(X_test_images, 'x_test_images_b2.pkl' )
    save_as_pickle(X_test_features, 'x_test_features_b2.pkl' )

    print("TASK B2. EYE COLOR DETECTION WITH RAW DATA ")
    model_b1(Y_train_labels, Y_test_labels, X_test_images, X_train_images, 'b2_raw_data')

    print("TASK B2. EYE COLOR DETECTION WITH FEATURES DATA ")
    model_b1(Y_train_labels, Y_test_labels, X_test_features, X_train_features, 'b2_feature_data')
    # -----------------------------------------------------------------------------------------------------

    # TASK B1:------------------------------------------------------------------------------------------------
    # Y_train_labels, X_train_images, X_train_features, Y_test_labels, X_test_images, X_test_features = get_data_b('face_shape')
    
    # save_as_pickle(Y_train_labels, 'y_train_labels_b1.pkl' )
    # save_as_pickle(X_train_images, 'x_train_images_b1.pkl' )
    # save_as_pickle(X_train_features, 'x_train_features_b1.pkl' )
    # save_as_pickle(Y_test_labels, 'y_test_labels_b1.pkl' )
    # save_as_pickle(X_test_images, 'x_test_images_b1.pkl' )
    # save_as_pickle(X_test_features, 'x_test_features_b1.pkl' )

    # print("TASK B1. EYE COLOR DETECTION WITH RAW DATA ")
    # model_b1(Y_train_labels, Y_test_labels, X_test_images, X_train_images, 'b1_raw_data')

    # print("TASK B1. EYE COLOR DETECTION WITH FEATURES DATA ")
    # model_b1(Y_train_labels, Y_test_labels, X_test_features, X_train_features, 'b1_feature_data')
    # --------------------------------------------------------------------------------------------------------------
    
    # # get data for task A1
    # Y_train_a1, Y_test_a1, X_test_a1, X_train_a1, train_images, test_images = get_data('gender')

    # # feature extraction for task A1 and A2
    # X_train_features, X_train_landmarks, training_images_shape = face_detection(train_images, None)

    # X_test_features, X_test_landmarks, _ = face_detection(test_images,training_images_shape)

    # # get data for task A2 
    # Y_train_a2, Y_test_a2, X_test_a2, X_train_a2, _, _  = get_data_a('smiling')
    # X_test_features, X_test_landmarks, _ = face_detection(test_images,training_images_shape)


    # print("TASK A1. GENDER PREDICTION MODEL WITH RAW DATA")
    # model_a1(Y_train_a1, Y_test_a1, X_test_a1, X_train_a1, "gender")
    # print("TASK A1. GENDER PREDICTION MODEL WITH FEATURES EXTRACTION")
    # model_a1(Y_train_a1, Y_test_a1, X_test_features, X_train_features, "gender")
    
    
    # print("TASK A2. SMILE PREDICTION MODEL WITH RAW DATA")
    # model_a2(Y_train_a2, Y_test_a2, X_test_a2, X_train_a2, "smiling")
    # print("TASK A2. SMILE PREDICTION MODEL WITH FEATURES EXTRACTION")
    # model_a2(Y_train_a2, Y_test_a2, X_test_features, X_train_features, "smiling")



if __name__ == "__main__":
    main()