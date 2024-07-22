
from helper_function import *
# Class names for CIFAR-10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def filter_classes(df, classes):
    """Filter dataset to include only the specified classes."""
    df_filtered = df[df['label'].isin(classes)]
    return df_filtered

def prepare_data_for_class_comparison(df, class_pairs):
    """Prepare data for comparison of specified class pairs."""
    for pair in class_pairs:
        class1, class2 = pair
        df_filtered = filter_classes(df, [class1, class2])
        X = df_filtered.drop('label', axis=1).values
        y = df_filtered['label'].values
        y = np.where(y == class1, 0, 1)  # Convert labels to binary
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Normalize and standardize data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        # KNN
        knn_model = KNeighborsClassifier(n_neighbors=3)
        knn_model.fit(X_train, y_train)
        knn_pred = knn_model.predict(X_test)
        knn_accuracy = accuracy_score(y_test, knn_pred)
        print(f"KNN ({class1} vs {class2}): {knn_accuracy:.2f}")

        # Logistic Regression
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        print(f"Logistic Regression ({class1} vs {class2}): {lr_accuracy:.2f}")

        # Decision Tree
        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train, y_train)
        dt_pred = dt_model.predict(X_test)
        dt_accuracy = accuracy_score(y_test, dt_pred)
        print(f"Decision Tree ({class1} vs {class2}): {dt_accuracy:.2f}")

if __name__ == '__main__':
    # Load and prepare data
    df = load_and_prepare_cifar_data()
    df['label'] = df['label'].apply(lambda x: CLASS_NAMES[x])  # Convert label indices to class names

    # Define class pairs for comparison
    class_pairs = [('airplane', 'ship'), ('automobile', 'ship'), ('truck', 'ship')]

    # Prepare data and evaluate models
    prepare_data_for_class_comparison(df, class_pairs)
