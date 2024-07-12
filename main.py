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

