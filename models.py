import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, optimizers
import tensorflow as tf

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

def SVM(X_train, X_val, X_test, y_train, y_val, y_test, ):
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

    best_val_accuracy = 0
    best_model_pred_val = None
    best_kernel = None
    best_C = None

    kernels = ['poly', 'rbf']
    C_values = [0.1, 1.0, 3.5, 5.0]

    for kernel in kernels:
        for C in C_values:
            # Initialize the SVM model
            model_svm = SVC(kernel=kernel, C=C)

            # Fit the model to the training data
            model_svm.fit(X_train, y_train)

            # Predict the validation and test data
            model_svm_pred_val = model_svm.predict(X_val)
            model_svm_pred_test = model_svm.predict(X_test)

            # Calculate and print accuracy
            val_accuracy = accuracy_score(y_val, model_svm_pred_val)
            test_accuracy = accuracy_score(y_test, model_svm_pred_test)

            print(
                f"Kernel = {kernel}, C = {C}: Validation Accuracy = {val_accuracy:.4f}, Test Accuracy = {test_accuracy:.4f}")

            # Keep track of the best model based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_pred_val = model_svm_pred_val
                best_kernel = kernel
                best_C = C

    print(f"Best Kernel: {best_kernel}, Best C: {best_C}, Best Validation Accuracy: {best_val_accuracy:.4f}")



def SVM_with_GridSearchCV(X_train, y_train, subsample_size=5000):
    """
    Trains and evaluates SVM models with different kernels and C values on the provided datasets using a subsample.

    Parameters:
    :param: X_train : Training data features.
    :param: y_train : Training data labels.
    :param: kernels : List of kernels to be used by SVM.
    :param: C_values : List of C values to be used by SVM.
    :param: subsample_size : Size of the subsample to use for faster model selection.

    :returns: The best model and its validation accuracy.
    """
    kernels = ['poly', 'rbf']
    C_values = [0.1, 1.0, 3.5, 5.0]

    # Subsample the data
    if len(X_train) > subsample_size:
        X_train_subsample = X_train[:subsample_size]
        y_train_subsample = y_train[:subsample_size]
    else:
        X_train_subsample = X_train
        y_train_subsample = y_train

    param_grid = {'C': C_values, 'kernel': kernels}

    # Initialize the SVM model with GridSearchCV
    grid_search = GridSearchCV(SVC(), param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_subsample, y_train_subsample)

    best_model = grid_search.best_estimator_
    best_val_accuracy = grid_search.best_score_
    best_params = grid_search.best_params_

    print(f"Best Kernel: {best_params['kernel']}, Best C: {best_params['C']}, Best Validation Accuracy: {best_val_accuracy:.4f}")

    return best_model, best_val_accuracy

def decision_tree(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Trains and evaluates a Decision Tree model on the provided datasets.

    Parameters:
    :param: X_train : Training data features.
    :param: X_val : Validation data features.
    :param: X_test : Test data features.
    :param: y_train : Training data labels.
    :param: y_val : Validation data labels.
    :param: y_test : Test data labels.

    Returns:
    np.ndarray: Predictions for the validation data using the best model.
    """
    # Initialize the Decision Tree model
    model_dt = DecisionTreeClassifier(random_state=42)

    # Fit the model to the training data
    model_dt.fit(X_train, y_train)

    # Predict the validation and test data
    model_dt_pred_val = model_dt.predict(X_val)
    model_dt_pred_test = model_dt.predict(X_test)

    # Calculate and print accuracy
    val_accuracy = accuracy_score(y_val, model_dt_pred_val)
    test_accuracy = accuracy_score(y_test, model_dt_pred_test)

    print(f"Validation Accuracy = {val_accuracy:.4f}, Test Accuracy = {test_accuracy:.4f}")

    return model_dt_pred_val

def CNN(X_train, X_val, X_test, y_train, y_val, y_test, input_shape, num_classes):
    """
    Trains and evaluates a Convolutional Neural Network (CNN) on the provided datasets.

    :param: X_train : Training data features.
    :param: X_val : Validation data features.
    :param: X_test : Test data features.
    :param: y_train : Training data labels.
    :param: y_val : Validation data labels.
    :param: y_test : Test data labels.
    :param: input_shape : Shape of the input data (height, width, channels).
    :param: num_classes : Number of classes.

    :returns: Predictions for the validation data using the trained CNN model.
    """

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Initialize the CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), verbose=1)

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print(f"Validation Accuracy = {val_accuracy:.4f}, Test Accuracy = {test_accuracy:.4f}")

    # Plotting validation vs test accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Predict on the validation data
    model_cnn_pred_val = model.predict(X_val)

    # Convert predictions back to label format
    model_cnn_pred_val = np.argmax(model_cnn_pred_val, axis=1)

    return model_cnn_pred_val


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, optimizers

def improved_CNN(x_train, x_val, y_train, y_val):
    """
    Trains and evaluates a revised Convolutional Neural Network (CNN) on CIFAR-10 dataset.

    :param x_train: Training data features.
    :param x_val: Validation data features.
    :param y_train: Training data labels.
    :param y_val: Validation data labels.

    :return: Validation accuracy of the trained CNN model.
    """
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)

    # Define the revised CNN model
    model_revised3 = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model_revised3.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

    # Print the model summary
    model_revised3.summary()

    # Train the model
    history_revised = model_revised3.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))

    # Plotting the training and validation accuracy over epochs
    plt.plot(history_revised.history['accuracy'], label='Train Accuracy')
    plt.plot(history_revised.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Make predictions on the validation data
    cnn_revised_val_pred = model_revised3.predict(x_val)

    # Convert the one-hot encoded predictions back to regular labels
    cnn_revised_val_pred = np.argmax(cnn_revised_val_pred, axis=1)
    y_val_labels = np.argmax(y_val, axis=1)

    # Calculate the accuracy score
    cnn_revised_val_acc = accuracy_score(y_val_labels, cnn_revised_val_pred)
    print('Revised CNN with Batch Normalization and fewer pooling layers validation accuracy:', cnn_revised_val_acc)

    return cnn_revised_val_acc
