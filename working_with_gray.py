from helper_function import *
from skimage.color import rgb2gray

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

def rgb2gray(rgb):
    """
    Convert RGB image to grayscale.
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def visualize_grayscale_images(X_gray, y, num_images=10):
    """
    Visualize grayscale images.
    """
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_gray[i].reshape(32, 32), cmap='gray')
        plt.title(CLASS_NAMES[y[i]])
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    df = load_and_prepare_cifar_data()

    # Convert images to grayscale
    X = df.drop(columns='label').values
    X_gray = np.array([rgb2gray(x.reshape(32, 32, 3)).flatten() for x in X])
    df_gray = pd.DataFrame(X_gray)
    df_gray['label'] = df['label']

    # Visualize the grayscale images
    # visualize_grayscale_images(X_gray, df_gray['label'].values)

    X_train, X_test, y_train, y_test = split_data(df_gray)

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
    X_train_gray = X_train.reshape(X_train.shape[0], -1)
    X_val_gray = X_val.reshape(X_val.shape[0], -1)
    X_test_gray = X_test.reshape(X_test.shape[0], -1)

    # Convert labels from a 2D array to 1D array
    y_train = np.squeeze(y_train)
    y_val = np.squeeze(y_val)
    y_test = np.squeeze(y_test)

    ###################### KNN ######################

    # # Run KNN and get predictions
    # model_knn_pred = KNN(X_train_gray, X_val_gray, y_train, y_val)
    #
    # # Plot the confusion matrix
    # plot_confusion_matrix(y_val, model_knn_pred, CLASS_NAMES)

    ###################### Logistic Regression ######################

    # # Train Logistic Regression and get predictions
    # model_lr_pred = logistic_regression(X_train_gray, X_val_gray, X_test_gray, y_train, y_val, y_test)
    #
    # # Plot the confusion matrix
    # plot_confusion_matrix(y_val, model_lr_pred, CLASS_NAMES)

    ############### Decision Tree ######################
    # Train Decision Tree and get predictions
    # model_dt_pred = decision_tree(X_train_gray, X_val_gray, X_test_gray, y_train, y_val, y_test)
    #
    # # Plot the confusion matrix
    # plot_confusion_matrix(y_val, model_dt_pred, CLASS_NAMES)