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

################## UPLOAD THE DATA ##################
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
response = requests.get(url, stream=True)

with open("cifar-10-python.tar.gz", "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            file.write(chunk)

with tarfile.open("cifar-10-python.tar.gz", "r:gz") as tar:
    tar.extractall()


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
