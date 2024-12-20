import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from helper_function import *

# Class names for CIFAR-10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to update labels
def update_labels(labels):
    # Label mapping
    label_mapping = {
        'airplane': 'vehicle',
        'automobile': 'vehicle',
        'ship': 'vehicle',
        'truck': 'vehicle',
        'bird': 'animal',
        'cat': 'animal',
        'deer': 'animal',
        'dog': 'animal',
        'frog': 'animal',
        'horse': 'animal'
    }
    return np.array([label_mapping[CLASS_NAMES[label]] for label in labels])

if __name__ == '__main__':
    df = load_and_prepare_cifar_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Update labels
    y_train = update_labels(y_train)
    y_test = update_labels(y_test)

    # Plot the distribution of the training labels
    # plot_class_distribution(y_train)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Normalize and standardize data
    normalize_data(X_train, X_val, X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Reshape data
    X_train_rgb = X_train.reshape(X_train.shape[0], -1)
    X_val_rgb = X_val.reshape(X_val.shape[0], -1)
    X_test_rgb = X_test.reshape(X_test.shape[0], -1)

    ###################### KNN ######################

    # # Run KNN and get predictions
    model_knn_pred = KNN(X_train_rgb, X_val_rgb, y_train, y_val)
    #
    # Convert categorical labels back to original labels for plotting confusion matrix
    y_val_decoded = [1 if label == 'animal' else 0 for label in y_val]
    # model_knn_pred_decoded = [1 if label == 'animal' else 0 for label in model_knn_pred]
    #
    # # Plot the confusion matrix
    # plot_confusion_matrix(y_val_decoded, model_knn_pred_decoded, ['vehicle', 'animal'])

    ###################### Logistic Regression ######################

    # # Train Logistic Regression and get predictions
    # model_lr_pred = logistic_regression(X_train_rgb, X_val_rgb, X_test_rgb, y_train, y_val, y_test)
    #
    # # Convert categorical labels back to original labels for plotting confusion matrix
    # model_lr_pred_decoded = [1 if label == 'animal' else 0 for label in model_lr_pred]
    #
    # # Plot the confusion matrix
    # plot_confusion_matrix(y_val_decoded, model_lr_pred_decoded, ['vehicle', 'animal'])

    ############### Decision Tree ######################
    # Train Decision Tree and get predictions
    # model_dt_pred = decision_tree(X_train_rgb, X_val_rgb, X_test_rgb, y_train, y_val, y_test)
    #
    # # Convert categorical labels back to original labels for plotting confusion matrix
    # model_dt_pred_decoded = [1 if label == 'animal' else 0 for label in model_dt_pred]
    #
    # # Plot the confusion matrix
    # plot_confusion_matrix(y_val_decoded, model_dt_pred_decoded, ['vehicle', 'animal'])




