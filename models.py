from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
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