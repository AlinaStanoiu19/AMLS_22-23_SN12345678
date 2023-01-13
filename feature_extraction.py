import cv2
import csv
import os
import numpy as np
import pandas as pd
import dlib
import matplotlib.pyplot as plt

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it to the format (x, y, w, h)
    # as we would normally do with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)

def shape_to_np(shape, dtype='int'):
    # the dlib face landmark detector will return a shape object containing
    # the 68 (x, y)-coordinates of the facial landmarks regions. this function
    # converts this object into a NumPy array

    coords = np.zeros((68,2), dtype=dtype)

    for i in range(0,68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    
    return coords

def face_detection(images, training_shape):
    # use dlib library to extract the 68 coordinates of the faces in each image
    # returns dict with images replaced with the cropped ones 
    # load the input images into a dictionaly with image names
    
    image_names = images.keys()
    no_faces = []
    image_shapes = []
    landmarks = []

    for img in image_names:    
        image = images[img].astype('uint8')
        # detect faces in the grayscale image
        rects = detector(image)
        num_faces = len(rects)

        if num_faces == 0:
            print(f"IN IMAGE {img} NO FACES WERE FOUND")
            no_faces.append(img)
            continue
        elif num_faces > 1:
            print(f"IN IMAGE {img} THERE ARE TOO MANY FACES FOUND")
            no_faces.append(img)
            continue
        
        shape = predictor(image, rects[0])
        shape = shape_to_np(shape)
        landmarks.append(shape)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = rect_to_bb(rects[0])
        cropped_image = image[(y- int(0.4*h)):( y + h + int(0.2*h)),(x- int(0.2*w)):(x + w + int(0.2*w))]
        # cropped_image = image[y:( y + h ),x:(x + w )]
        image_shapes.append((cropped_image.shape[0],cropped_image.shape[1]))
        images[img] = cropped_image

    print(f"no of images without faces: {len(no_faces)}")
    if training_shape == None:
        image_shapes = np.array(image_shapes)
        max_h = max(image_shapes[:,0])
        max_w = max(image_shapes[:,1])
        print(f"the images will be reshaped to: {(max_h, max_w)}")
        X_features = []
        for img in image_names:
            resized_image = cv2.resize(images[img],(max_h, max_w))
            # cv2.imwrite(img +'resized.jpg', resized_image)
            images[img] = resized_image
            X_features.append(resized_image)
    else: 
       print(f"the shape of the training images is: {training_shape}")
       X_features = []
       (max_h, max_w )= training_shape
       for img in image_names:
            resized_image = cv2.resize(images[img],(max_h, max_w))
            # cv2.imwrite(img +'resized.jpg', resized_image)
            images[img] = resized_image
            X_features.append(resized_image)
    X_features = np.array(X_features)
    landmarks= np.array(landmarks)
    print(X_features.shape)
    return X_features, landmarks, (max_h,max_w)

def cartoon_face_detection(images):
    image_names = images.keys()
    landmarks_dict = dict.fromkeys(image_names)
    gray_images = dict.fromkeys(image_names)
    no_faces_list = []

    for img in image_names:
        image = images[img].astype('uint8')
        gray_images[img] = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        rects = detector(gray_images[img])

        num_faces = len(rects)
        if num_faces != 1: 
            print(f"IN IMAGE {str(img)} NO FACES WERE FOUND")
            no_faces_list.append(img)
            landmarks_dict[img] = -1
            continue
        
        landmarks = predictor(gray_images[img],rects[0])
        landmarks = shape_to_np(landmarks)
        landmarks_dict[img] = landmarks

    return images, gray_images, landmarks_dict, no_faces_list

def get_average_face_center(landmarks_dict, no_faces_list ):
    center_landmark = []
    list_faces = [ele for ele in landmarks_dict.keys() if ele not in no_faces_list]
    
    for x in  list_faces:
        center_landmark.append(landmarks_dict[x][28][1])
    
    remove_no_face = [i for i in center_landmark if i != -1]
    return sum(remove_no_face)/len(remove_no_face)


def face_shape_feature_extraction(input_images):
    _, gray_images, landmarks_dict, no_faces_list = cartoon_face_detection(input_images)
    image_names = gray_images.keys()
    features = dict.fromkeys(image_names)
    for img in image_names:
        # plt.imshow(gray_images[img],cmap='gray')
        if img in no_faces_list:
            landmarks_dict[img] = get_average_face_center(landmarks_dict, no_faces_list)
            center_landmark = int(landmarks_dict[img])
        else: 
            center_landmark = landmarks_dict[img][28][1]
        features[img] = gray_images[img][center_landmark:(center_landmark+150),50:450]
        h, w = features[img].shape
        
        for y in range(w):
            for x in range(h):
                if features[img][x,y] > 80:
                    features[img][x,y] = 255
    
    return features

def remove_sunglasses(image):
    glasses = False
    hist,bin = np.histogram(image.ravel(),256,[0,255]) 
    for i in range(0,30):
        print(hist[i])
        if hist[i] > 2500:
            glasses = True
    return glasses


def get_average_side(landmarks_dict, no_faces_list ):
    top = []
    bottom = []
    left =[]
    right = []
    list_faces = [ele for ele in landmarks_dict.keys() if ele not in no_faces_list]
    
    for x in  list_faces:
        top.append(landmarks_dict[x][18][1])
        bottom.append(landmarks_dict[x][29][1])
        left.append(landmarks_dict[x][0][0])
        right.append(landmarks_dict[x][16][0])
    
    # remove_no_face = [i for i in top if i != -1]
    return int(sum(top)/len(top)), int(sum(bottom)/len(bottom)),int(sum(left)/len(left)),int(sum(right)/len(right))

def eye_feature_extraction(input_images):
    images, gray_images, landmarks_dict, no_faces_list = cartoon_face_detection(input_images)
    image_names = gray_images.keys()
    features = dict.fromkeys(image_names)
    features_gray = dict.fromkeys(image_names)
    sunglasses_images = []
    for img in image_names:
        if img in no_faces_list:
            top, bottom, left, right = get_average_side(landmarks_dict, no_faces_list)
        else: 
            top = landmarks_dict[img][18][1]
            bottom = landmarks_dict[img][29][1]
            left = landmarks_dict[img][0][0]
            right = landmarks_dict[img][16][0]
        features_gray[img] = gray_images[img][top:bottom,left:right]
        if remove_sunglasses(features_gray[img]):
            sunglasses_images.append(img)
        else:
            features[img] = images[img][top:bottom,left:right]   


    
    return features, sunglasses_images
