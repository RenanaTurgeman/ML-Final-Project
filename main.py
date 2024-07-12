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

# Download the CIFAR-10 dataset
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
response = requests.get(url, stream=True)

with open("cifar-10-python.tar.gz", "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            file.write(chunk)

with tarfile.open("cifar-10-python.tar.gz", "r:gz") as tar:
    tar.extractall()

# Load the data from the extracted files
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train_data = []
train_labels = []

for i in range(1, 6):
    batch = unpickle(f'cifar-10-batches-py/data_batch_{i}')
    train_data.append(batch[b'data'])
    train_labels.append(batch[b'labels'])

train_data = np.vstack(train_data).astype(np.uint8)  # Ensure data is uint8
train_labels = np.hstack(train_labels)

# Convert to DataFrame
train_data_df = pd.DataFrame(train_data)
train_labels_df = pd.Series(train_labels, name='label')

# CIFAR-10 class names
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Map numerical labels to class names
train_labels_df = train_labels_df.map(lambda x: label_names[x])

# Function to plot the distribution of labels in the training data
def distribution_of_labels():
    """
    Function to plot the distribution of labels in the training data
    :return: plot of the distribution of labels in the training data
    """
    sns.countplot(x=train_labels_df, order=label_names)
    plt.title('Distribution of Labels in Training Data')
    plt.xlabel('Class Names')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

def example_of_images():
    # Display examples of images in the data
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        img = train_data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        ax.imshow(img)
        ax.set_title(f"Label: {train_labels_df.iloc[i]}")
        ax.axis('off')

    plt.show()

def understanding_the_data():
    distribution_of_labels()
    example_of_images()

understanding_the_data()
