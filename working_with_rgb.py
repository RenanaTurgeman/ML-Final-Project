"""
CIFAR-10 dataset.
For Image classification.
Link to the dataset: https://www.kaggle.com/datasets/fedesoriano/cifar10-python-in-csv?resource=download

Programmers: Renana Turgeman, Ofir Shitrit.
Since: 2024-07
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import tarfile
import pickle
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from models import *
from helper_function import *

# Class names for CIFAR-10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

################## UPLOAD THE DATA ##################
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
response = requests.get(url, stream=True)

with open("cifar-10-python.tar.gz", "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            file.write(chunk)

with tarfile.open("cifar-10-python.tar.gz", "r:gz") as tar:
    tar.extractall()



if __name__ == '__main__':
    df = load_and_prepare_cifar_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Print the shapes of the resulting datasets
    # print(f"X_train shape: {X_train.shape}")
    # print(f"X_test shape: {X_test.shape}")
    # print(f"y_train shape: {y_train.shape}")
    # print(f"y_test shape: {y_test.shape}")

    #| output:
    # X_train shape: (40000, 3072)
    # X_test shape: (10000, 3072)
    # y_train shape: (40000,)
    # y_test shape: (10000,)
    # |#

    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # Normalize the data
    normalize_data(X_train, X_val, X_test)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Reshape the data
    X_train_rgb = X_train.reshape(X_train.shape[0], -1)
    X_val_rgb = X_val.reshape(X_val.shape[0], -1)
    X_test_rgb = X_test.reshape(X_test.shape[0], -1)

    # Convert labels from a 2D array to 1D array
    y_train = np.squeeze(y_train)
    y_val = np.squeeze(y_val)
    y_test = np.squeeze(y_test)

    ###################### KNN ######################

    # # Run KNN and get predictions
    # model_knn_pred = KNN(X_train_rgb, X_val_rgb, y_train, y_val)
    #
    # # Plot the confusion matrix
    # plot_confusion_matrix(y_val, model_knn_pred, CLASS_NAMES)

    ###################### Logistic Regression ######################

    # # Train Logistic Regression and get predictions
    # model_lr_pred = logistic_regression(X_train_rgb, X_val_rgb, X_test_rgb, y_train, y_val, y_test)
    #
    # # Plot the confusion matrix
    # plot_confusion_matrix(y_val, model_lr_pred, CLASS_NAMES)

    ############### Decision Tree ######################
    # Train Decision Tree and get predictions
    # model_dt_pred = decision_tree(X_train_rgb, X_val_rgb, X_test_rgb, y_train, y_val, y_test)
    #
    # # Plot the confusion matrix
    # plot_confusion_matrix(y_val, model_dt_pred, CLASS_NAMES)

    ###################### CNN ######################
    #
    # # Reshape the data for CNN
    # X_train_cnn = X_train.reshape(X_train.shape[0], 32, 32, 3)
    # X_val_cnn = X_val.reshape(X_val.shape[0], 32, 32, 3)
    # X_test_cnn = X_test.reshape(X_test.shape[0], 32, 32, 3)
    #
    # # Define the input shape and number of classes
    # input_shape = (32, 32, 3)
    # num_classes = 10

    # # Train CNN and get predictions
    # model_cnn_pred = CNN(X_train_cnn, X_val_cnn, X_test_cnn, y_train, y_val, y_test, input_shape, num_classes)
    #
    # # Plot the confusion matrix
    # plot_confusion_matrix(y_val, model_cnn_pred, CLASS_NAMES)

    ###################### IMPROVED CNN ######################
    # # Reshape the data for CNN
    # X_train_cnn = X_train.reshape(X_train.shape[0], 32, 32, 3)
    # X_val_cnn = X_val.reshape(X_val.shape[0], 32, 32, 3)
    # X_test_cnn = X_test.reshape(X_test.shape[0], 32, 32, 3)
    #
    # # Convert labels to categorical
    # y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=10)
    # y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=10)
    # y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=10)
    #
    # # Train improved CNN and get results
    # model_revised3, history_revised, conf_matrix_revised, cnn_revised_val_acc = improved_CNN(X_train_cnn, y_train_cat, X_val_cnn, y_val_cat)
    #
    # # Plot the confusion matrix (assuming you have a function plot_confusion_matrix)
    # # plot_confusion_matrix(y_val, cnn_revised_val_pred, CLASS_NAMES)

    ###################### MORE IMPROVED CNN ######################
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    # Train the improved CNN
    model, history, conf_matrix, accuracy = more_improved_CNN(X_train, y_train, X_val, y_val)