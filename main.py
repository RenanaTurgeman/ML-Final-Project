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


################ HELPER FUNCTIONS ##################
def load_cifar_batch(file_path):
    """
    Loads a CIFAR-10 batch file and returns the data as a dictionary.

    :param: file_path : The path to the CIFAR-10 batch file.

    :return: dict:  A dictionary containing the data and labels from the CIFAR-10 batch file.
                    The dictionary contains the following keys:
                        - b'data': A numpy array of shape (10000, 3072) containing the image data.
                        - b'labels': A list of length 10000 containing the labels for the images.
                        - Other metadata keys such as b'batch_label' and b'filenames'.
    """
    with open(file_path, 'rb') as file:
        data_dict = np.load(file, encoding='bytes', allow_pickle=True)
    return data_dict


def normalize_data(train, val, test):
    """
    Normalizes the training, validation, and test datasets by scaling the pixel values to be between 0 and 1.

    :param: train :  The training dataset.
    :param: val : The validation dataset.
    :param: test : The test dataset.

    This function modifies the input datasets in place by dividing each element by 255.0.
    """
    train = train / 255.0
    val = val / 255.0
    test = test / 255.0
    return train, val, test


def display_cifar_images(num, dataset_path='./cifar-10-batches-py'):
    """
    Loads and displays images from a CIFAR-10 batch file.

    :param: num : If 0, the images will be upscaled to 128x128. Otherwise, the images will be displayed in their original size (32x32).
    :param: dataset_path : optional - The path to the CIFAR-10 dataset folder. Default is './cifar-10-batches-py'.

    The function loads a batch of CIFAR-10 training data, reshapes the pixel data into image format,
    optionally upscales the images, and displays the first 10 images along with their labels.
    """
    # Load CIFAR-10 batch data
    batch_path = os.path.join(dataset_path, 'data_batch_1')
    batch = load_cifar_batch(batch_path)

    # Extract pixel data and labels
    pixel_data = batch[b'data']
    labels = batch[b'labels']

    # Reshape pixel data to image format
    images = pixel_data.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)

    # Optionally upscale images
    if num == 0:
        images = [np.array(Image.fromarray(img).resize((128, 128), Image.BILINEAR)) for img in images]

    # Plot the first 10 images and their labels
    fig, axes = plt.subplots(1, 10, figsize=(15, 3))

    for i in range(10):
        img = images[i]
        label = labels[i]
        axes[i].imshow(img)
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# todo - check this function
def load_and_prepare_cifar_data(dataset_path='./cifar-10-batches-py'):
    """
    Loads and preprocesses the CIFAR-10 dataset from the specified directory.

    :param: dataset_path :  optional -
        The path to the CIFAR-10 dataset folder. Default is './cifar-10-batches-py'.

    :returns: pd.DataFrame
        A DataFrame containing the flattened image data and their corresponding labels.
    """
    data_list = []

    # Loop through each batch file and load the data
    for i in range(1, 6):
        batch_file_path = os.path.join(dataset_path, f'data_batch_{i}')
        batch_data = load_cifar_batch(batch_file_path)

        # Extract the pixel data and labels from the batch
        pixel_data = batch_data[b'data']
        labels = batch_data[b'labels']

        # Reshape the pixel data into image format
        images = pixel_data.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)

        # Flatten the images as feature vectors
        num_images = images.shape[0]
        image_size = images.shape[1] * images.shape[2] * images.shape[3]
        flattened_images = images.reshape(num_images, image_size)

        # Create a DataFrame with flattened images and labels
        df_batch = pd.DataFrame(flattened_images)
        df_batch['label'] = labels

        # Append the batch data to the data_list
        data_list.append(df_batch)

    # Concatenate all batch DataFrames into a single DataFrame
    df = pd.concat(data_list, ignore_index=True)

    return df


# df = load_and_prepare_cifar_data()
# print(df)

def split_data(df, test_size=0.2):
    """
    Splits the DataFrame into training and testing datasets.

    :param: df : pd.DataFrame
        The DataFrame containing the flattened image data and their corresponding labels.
    :param: test_size : float, optional
        The proportion of the dataset to include in the test split. Default is 0.2.

    :returns: tuple
        A tuple containing four elements:
            - X_train: Training feature vectors.
            - X_test: Testing feature vectors.
            - y_train: Training labels.
            - y_test: Testing labels.
    """
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plots the confusion matrix for the given true labels and predicted labels.

    Parameters:
    y_true: True labels.
    y_pred: Predicted labels.
    class_names: List of class names.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


########## SIMPLE MODELS ##########

def KNN(train, val, y_train, y_val):
    """
    Performs K-Nearest Neighbors classification on the provided training and validation datasets.

    :param train: The training feature vectors.
    :param val: The validation feature vectors.
    :param y_train: The labels for the training feature vectors.
    :param y_val: The labels for the validation feature vectors.

    :return: The predictions for the validation set.
    """
    accuracy = []

    # Test odd values for n_neighbors from 1 to 15
    for i in range(1, 15, 2):
        # Initialize the KNN model
        model_knn = KNeighborsClassifier(n_neighbors=i)

        # Fit the model on the training data
        model_knn.fit(train, y_train)

        # Predict on the validation data
        model_knn_pred = model_knn.predict(val)

        # Calculate accuracy
        acc_knn = accuracy_score(y_val, model_knn_pred)
        accuracy.append(round(acc_knn, 3) * 100)

    # Find the best accuracy and corresponding number of neighbors
    best_index = accuracy.index(max(accuracy)) + 1
    print(f"The Accuracy score for {best_index} nearest neighbors is: {max(accuracy)}%")

    # Plot the accuracy against the number of neighbors
    plt.plot(range(1, 15, 2), accuracy)
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy (%)')
    plt.title('KNN Accuracy vs. Number of Neighbors')
    plt.grid()
    plt.show()

    return model_knn_pred


def logistic_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Trains and evaluates Logistic Regression models with different max_iter values on the provided datasets.

    Parameters:
    :param: X_train : Training data features.
    :param: X_val : Validation data features.
    :param: X_test : Test data features.
    :param: y_train : Training data labels.
    :param: y_val : Validation data labels.
    :param: y_test : Test data labels.

    :returns: Predictions for the validation data using the best model.
    """
    # Define a list of max_iter values
    max_iter_values = [100, 500, 1000, 1500, 2000]
    best_val_accuracy = 0
    best_model_pred_val = None

    for iter in max_iter_values:
        # Initialize the Logistic Regression model
        model_lr = LogisticRegression(max_iter=iter, solver='lbfgs')

        # Fit the model to the training data
        model_lr.fit(X_train, y_train)

        # Predict the validation and test data
        model_lr_pred_val = model_lr.predict(X_val)
        model_lr_pred_test = model_lr.predict(X_test)

        # Calculate and print accuracy
        val_accuracy = accuracy_score(y_val, model_lr_pred_val)
        test_accuracy = accuracy_score(y_test, model_lr_pred_test)

        print(f"iter = {iter}: Validation Accuracy = {val_accuracy:.4f}, Test Accuracy = {test_accuracy:.4f}")

        # Keep track of the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_pred_val = model_lr_pred_val

    return best_model_pred_val


def svm_classifier(X_train, X_val, X_test, y_train, y_val, y_test, ):
    """
    Trains and evaluates SVM models with different kernels and C values on the provided datasets.

    Parameters:
    :param: X_train : Training data features.
    :param: X_val : Validation data features.
    :param: X_test : Test data features.
    :param: y_train : Training data labels.
    :param: y_val : Validation data labels.
    :param: y_test : Test data labels.
    :param: kernels : List of kernels to be used by SVM.
    :param: C_values : List of C values to be used by SVM.

    :returns: None
    """
    # best_val_accuracy = 0
    # best_model_pred_val = None
    # best_kernel = None
    # best_C = None
    #
    # kernels = ['poly', 'rbf']
    # C_values = [0.1, 1.0, 3.5, 5.0]
    #
    # for kernel in kernels:
    #     for C in C_values:
    #         # Initialize the SVM model
    #         model_svm = SVC(kernel=kernel, C=C)
    #
    #         # Fit the model to the training data
    #         model_svm.fit(X_train, y_train)
    #
    #         # Predict the validation and test data
    #         model_svm_pred_val = model_svm.predict(X_val)
    #         model_svm_pred_test = model_svm.predict(X_test)
    #
    #         # Calculate and print accuracy
    #         val_accuracy = accuracy_score(y_val, model_svm_pred_val)
    #         test_accuracy = accuracy_score(y_test, model_svm_pred_test)
    #
    #         print(
    #             f"Kernel = {kernel}, C = {C}: Validation Accuracy = {val_accuracy:.4f}, Test Accuracy = {test_accuracy:.4f}")
    #
    #         # Keep track of the best model based on validation accuracy
    #         if val_accuracy > best_val_accuracy:
    #             best_val_accuracy = val_accuracy
    #             best_model_pred_val = model_svm_pred_val
    #             best_kernel = kernel
    #             best_C = C
    #
    # print(f"Best Kernel: {best_kernel}, Best C: {best_C}, Best Validation Accuracy: {best_val_accuracy:.4f}")
    #
    # # Plot the confusion matrix for the best model
    # plot_confusion_matrix(y_val, best_model_pred_val, CLASS_NAMES)
    accuracy = []
    Kernel = ['poly', 'rbf']
    c = [0.1, 1.0, 2.0, 5.0]
    accuracy_scores = {kernel: {'train': [], 'val': []} for kernel in Kernel}

    for i in range(len(Kernel)):
        for j in range(len(c)):
            # Initialize the SVM classifier
            SVM = SVC(max_iter=10000, kernel=Kernel[i], C=c[j])

            # Fit the SVM classifier on the training data
            SVM.fit(X_train, y_train)

            # Make predictions on the validation data
            val_pred = SVM.predict(X_val)
            model_svm_t = SVM.predict(X_train)

            # Calculate the accuracy score
            svm_acc = accuracy_score(y_val, val_pred)
            acc_svm_t = accuracy_score(y_train, model_svm_t)
            accuracy_scores[Kernel[i]]['val'].append(svm_acc)
            accuracy_scores[Kernel[i]]['train'].append(acc_svm_t)

            print("Validation:")
            print(Kernel[i], 'kernel with', c[j], 'Regularization parameter gave score of:', svm_acc)
            print("Train:")
            print(Kernel[i], 'kernel with', c[j], 'Regularization parameter gave score of:', acc_svm_t)

        # Create a plot for each kernel (validation accuracy)
        plt.figure(figsize=(10, 6))
        plt.plot(c, accuracy_scores[Kernel[i]]['val'], marker='o', label=Kernel[i] + ' (Validation)')
        plt.plot(c, accuracy_scores[Kernel[i]]['train'], marker='x', label=Kernel[i] + ' (Train)')
        plt.title('SVM Accuracy for ' + Kernel[i] + ' Kernel')
        plt.xlabel('Regularization Parameter (C)')
        plt.ylabel('Accuracy')
        plt.xticks(c)
        plt.legend()
        plt.grid(True)
        plt.show()
    return val_pred

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

    ###################### SVM ######################

    # Train SVM and get predictions
    model_svm_pred = svm_classifier(X_train_rgb, X_val_rgb, X_test_rgb, y_train, y_val, y_test)

    # Plot the confusion matrix
    plot_confusion_matrix(y_val, model_svm_pred, CLASS_NAMES)
