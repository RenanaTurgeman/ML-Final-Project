import numpy as np
from keras.src.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.src.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import KNeighborsClassifier
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
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, optimizers
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

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



def improved_CNN(x_train, y_train, x_val, y_val, input_shape=(32, 32, 3), num_classes=10, epochs=10, batch_size=64):
    # The revised model with batch normalization and fewer pooling layers
    model_revised3 = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
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
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model_revised3.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    # Print the model summary
    model_revised3.summary()

    # Train the model
    history_revised = model_revised3.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))

    # Make predictions on the validation data
    cnn_revised_val_pred = model_revised3.predict(x_val)

    # Convert the one-hot encoded back to regular labels
    cnn_revised_val_pred = np.argmax(cnn_revised_val_pred, axis=1)
    y_val_labels = np.argmax(y_val, axis=1)

    # Calculate the confusion matrix
    conf_matrix_revised = confusion_matrix(y_val_labels, cnn_revised_val_pred)

    # Calculate the accuracy score
    cnn_revised_val_acc = accuracy_score(y_val_labels, cnn_revised_val_pred)
    print('CNN accuracy:', cnn_revised_val_acc)

    return model_revised3, history_revised, conf_matrix_revised, cnn_revised_val_acc


def more_improved_CNN(x_train, y_train, x_val, y_val, input_shape=(32, 32, 3), num_classes=10, epochs=50, batch_size=64):
    # Reshape the input data to the correct format
    x_train = x_train.reshape(-1, 32, 32, 3)
    x_val = x_val.reshape(-1, 32, 32, 3)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    # Revised model
    model_revised3 = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
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
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model_revised3.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

    # Train the model with data augmentation
    history_revised = model_revised3.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                                         epochs=epochs,
                                         validation_data=(x_val, y_val),
                                         callbacks=[early_stopping, model_checkpoint, reduce_lr])

    # Make predictions on the validation data
    cnn_revised_val_pred = model_revised3.predict(x_val)

    # Convert the one-hot encoded back to regular labels
    cnn_revised_val_pred = np.argmax(cnn_revised_val_pred, axis=1)
    y_val_labels = np.argmax(y_val, axis=1)

    # Calculate the confusion matrix
    conf_matrix_revised = confusion_matrix(y_val_labels, cnn_revised_val_pred)

    # Calculate the accuracy score
    cnn_revised_val_acc = accuracy_score(y_val_labels, cnn_revised_val_pred)
    print('CNN accuracy:', cnn_revised_val_acc)

    return model_revised3, history_revised, conf_matrix_revised, cnn_revised_val_acc

