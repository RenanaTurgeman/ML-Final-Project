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