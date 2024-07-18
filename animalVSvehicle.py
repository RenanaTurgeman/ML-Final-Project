import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from helper_function import *

# Class names for CIFAR-10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to update labels
def update_labels(labels):
    # Label mapping
    label_mapping = {
        'airplane': 'vehicle',
        'automobile': 'vehicle',
        'ship': 'vehicle',
        'truck': 'vehicle',
        'bird': 'animal',
        'cat': 'animal',
        'deer': 'animal',
        'dog': 'animal',
        'frog': 'animal',
        'horse': 'animal'
    }
    return np.array([label_mapping[CLASS_NAMES[label]] for label in labels])

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

if __name__ == '__main__':
    df = load_and_prepare_cifar_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Update labels
    y_train = update_labels(y_train)
    y_test = update_labels(y_test)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Normalize and standardize data
    normalize_data(X_train, X_val, X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Reshape data
    X_train_rgb = X_train.reshape(X_train.shape[0], -1)
    X_val_rgb = X_val.reshape(X_val.shape[0], -1)
    X_test_rgb = X_test.reshape(X_test.shape[0], -1)

    ###################### KNN ######################

    # Run KNN and get predictions
    model_knn_pred = KNN(X_train_rgb, X_val_rgb, y_train, y_val)

    # Plot the confusion matrix
    plot_confusion_matrix(y_val, model_knn_pred, update_labels(CLASS_NAMES))

    ###################### Logistic Regression ######################

    # # Train Logistic Regression and get predictions
    # model_lr_pred = logistic_regression(X_train_rgb, X_val_rgb, X_test_rgb, y_train, y_val, y_test)
    #
    # # Plot the confusion matrix
    # plot_confusion_matrix(y_val, model_lr_pred, CLASS_NAMES)

    ###################### SVM ######################

    # # Train SVM and get predictions
    # model_svm_pred = SVM(X_train_rgb, X_val_rgb, X_test_rgb, y_train, y_val, y_test)
    #
    # # Plot the confusion matrix
    # plot_confusion_matrix(y_val, model_svm_pred, CLASS_NAMES)

    ###################### SVM WITH GridSearchCV ######################

    # # Train SVM and get the best model
    # best_svm_model, best_val_accuracy = SVM_with_GridSearchCV(X_train_rgb, y_train)
    #
    # # Predict on the validation data using the best model
    # model_svm_pred_val = best_svm_model.predict(X_val_rgb)
    #
    # # Plot the confusion matrix for the best model
    # plot_confusion_matrix(y_val, model_svm_pred_val, CLASS_NAMES)

    ############### Decision Tree ######################
    # Train Decision Tree and get predictions
    # model_dt_pred = decision_tree(X_train_rgb, X_val_rgb, X_test_rgb, y_train, y_val, y_test)
    #
    # # Plot the confusion matrix
    # plot_confusion_matrix(y_val, model_dt_pred, CLASS_NAMES)

    ###################### CNN ######################
    #
    # # Reshape the data for CNN
    # X_train_cnn = X_train.reshape(X_train.shape[0], 32, 32, 3)
    # X_val_cnn = X_val.reshape(X_val.shape[0], 32, 32, 3)
    # X_test_cnn = X_test.reshape(X_test.shape[0], 32, 32, 3)
    #
    # # Define the input shape and number of classes
    # input_shape = (32, 32, 3)
    # num_classes = 10

    # # Train CNN and get predictions
    # model_cnn_pred = CNN(X_train_cnn, X_val_cnn, X_test_cnn, y_train, y_val, y_test, input_shape, num_classes)
    #
    # # Plot the confusion matrix
    # plot_confusion_matrix(y_val, model_cnn_pred, CLASS_NAMES)

    ###################### IMPROVED CNN ######################
    # # Reshape the data for CNN
    # X_train_cnn = X_train.reshape(X_train.shape[0], 32, 32, 3)
    # X_val_cnn = X_val.reshape(X_val.shape[0], 32, 32, 3)
    # X_test_cnn = X_test.reshape(X_test.shape[0], 32, 32, 3)
    #
    # # Convert labels to categorical
    # y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=10)
    # y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=10)
    # y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=10)
    #
    # # Train improved CNN and get results
    # model_revised3, history_revised, conf_matrix_revised, cnn_revised_val_acc = improved_CNN(X_train_cnn, y_train_cat, X_val_cnn, y_val_cat)
    #
    # # Plot the confusion matrix (assuming you have a function plot_confusion_matrix)
    # # plot_confusion_matrix(y_val, cnn_revised_val_pred, CLASS_NAMES)

    ###################### MORE IMPROVED CNN ######################
    # # Convert labels to categorical
    # y_train = tf.keras.utils.to_categorical([1 if label == 'animal' else 0 for label in y_train], num_classes=2)
    # y_val = tf.keras.utils.to_categorical([1 if label == 'animal' else 0 for label in y_val], num_classes=2)
    # y_test = tf.keras.utils.to_categorical([1 if label == 'animal' else 0 for label in y_test], num_classes=2)
    #
    # # Train models with new data
    # model, history, conf_matrix, accuracy = more_improved_CNN(X_train, y_train, X_val, y_val)
    #
    # # Display results (you can add functions to display additional results such as confusion matrix, performance report, etc.)
    # print(f"Confusion Matrix:\n{conf_matrix}")
    # print(f"Accuracy: {accuracy}")
