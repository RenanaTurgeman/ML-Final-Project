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