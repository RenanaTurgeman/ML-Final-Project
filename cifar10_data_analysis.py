"""
analysis of CIFAR-10 dataset.
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

#################### LOAD THE DATA ###########################

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


################### UNDERSTANDING THE DATA ##################################
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


def basic_statistics():
    """
    Function to display basic statistics of the dataset
    """
    print(train_data_df.describe())


def random_sample_images():
    """
    Function to display a random sample of images from the dataset
    """
    sample_indices = np.random.choice(train_data_df.index, size=10, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        img = train_data_df.iloc[sample_indices[i]].values.reshape(3, 32, 32).transpose(1, 2, 0)
        ax.imshow(img)
        ax.set_title(f"Label: {train_labels_df.iloc[sample_indices[i]]}")
        ax.axis('off')
    plt.show()


def pixel_value_distribution():
    """
    Function to plot the distribution of pixel values in the dataset
    """
    pixel_values = train_data_df.values.flatten()
    plt.hist(pixel_values, bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Pixel Values')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()


def normalize_images():
    """
    Function to normalize the images and display a random sample of normalized images
    """
    normalized_data = train_data_df / 255.0
    sample_indices = np.random.choice(normalized_data.index, size=10, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        img = normalized_data.iloc[sample_indices[i]].values.reshape(3, 32, 32).transpose(1, 2, 0)
        ax.imshow(img)
        ax.set_title(f"Label: {train_labels_df.iloc[sample_indices[i]]}")
        ax.axis('off')
    plt.show()


def pixel_statistics_per_channel():
    """
    Function to display the mean and standard deviation of pixel values per channel (RGB)
    """
    means = train_data_df.mean(axis=0).values.reshape(3, 32, 32).mean(axis=(1, 2))
    stds = train_data_df.std(axis=0).values.reshape(3, 32, 32).std(axis=(1, 2))
    print("Mean pixel values per channel (R, G, B):", means)
    print("Standard deviation of pixel values per channel (R, G, B):", stds)


def understanding_the_data():
    distribution_of_labels()
    example_of_images()
    basic_statistics()
    random_sample_images()
    pixel_value_distribution()
    normalize_images()
    pixel_statistics_per_channel()


# Call the understanding_the_data function to perform all analyses
understanding_the_data()
