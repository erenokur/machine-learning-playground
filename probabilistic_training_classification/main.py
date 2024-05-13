import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def calculate_probabilistic_classification_accuracy(model, features, labels):
    """
    Calculates the accuracy of a probabilistic classifier on a dataset.

    Args:
        model: Trained probabilistic classifier.
        features: Input data.
        labels: True class labels.

    Returns:
        Accuracy of the classifier.
    """
    if len(features) != len(labels):
        raise ValueError("Length of features and labels must be the same.")
    if len(features) == 0 or len(labels) == 0:
        raise ValueError("Features and labels must not be empty.")
    if model is None:
        raise ValueError("Model must be instantiated.")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train the model (assuming model is not already trained)
    model.fit(X_train, y_train)

    # Predict probabilities for each class
    predicted_probabilities = model.predict_proba(X_test)

    # Predict the class with the highest probability
    predicted_classes = np.argmax(predicted_probabilities, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predicted_classes)

    return accuracy

def main():
    # Load the Iris dataset
    iris = load_iris()

    # Train the Naive Bayes classifier
    model = GaussianNB()
    accuracy = calculate_probabilistic_classification_accuracy(model, iris.data, iris.target)

    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()

