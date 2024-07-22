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
from sklearn.decomposition import PCA  # Import PCA
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

    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # Normalize the data
    normalize_data(X_train, X_val, X_test)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Apply PCA
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    X_train_pca = pca.fit_transform(X_train)

    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel("Number of Features")
    # plt.ylabel("The Variance")
    # plt.show()

    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    # Visualize eigenvalues to show noise reduction
    eigenvalues = pca.explained_variance_
    plt.figure(figsize=(8, 6))
    plt.plot(eigenvalues, marker='o')
    plt.xlabel("Principal Component Index")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalues of Principal Components")
    plt.grid(True)
    plt.show()

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
