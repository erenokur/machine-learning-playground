from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def calculate_error_rate(model, features, labels):
    """
    Calculates classification error rate on a dataset.

    Args:
        model: Trained classification model.
        features: Input data.
        labels: True class labels.

    Returns:
        Error rate (1 - accuracy).
    """
    if len(features) != len(labels):
        raise ValueError("Length of features and labels must be the same.")
    if len(features) == 0 or len(labels) == 0:
        raise ValueError("Features and labels must not be empty.")
    if model is None:
        raise ValueError("Model must be instantiated.")

    # Split data into training and testing sets
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train the model (assuming model is not already trained)
    model.fit(features_train, labels_train)

    # Make predictions on the test set
    predictions = model.predict(features_test)

    # Calculate accuracy and error rate
    accuracy = accuracy_score(labels_test, predictions)
    error_rate = 1 - accuracy

    return error_rate

def main():
    # Example data
    features = np.array([
        [10, 2, 5],
        [3, 8, 1],
        [5, 7, 9],
        [1, 3, 5],
        [2, 4, 6],
        [8, 10, 12],
        [11, 13, 15], 
        [14, 16, 18],  
        [17, 19, 21],  
        [20, 22, 24]  
    ])

    # Original labels
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  

    # Add noise to labels (flip some of the labels to simulate noisy data)
    np.random.seed(42)  # for reproducibility
    noise = np.random.randint(0, 2, size=labels.shape)
    labels_noisy = labels ^ noise  # XOR operation to flip some of the labels

    # Create a model
    model = RandomForestClassifier(n_estimators=100)

    # Example usage (assuming you have your data and model)
    error_rate = calculate_error_rate(model, features, labels_noisy)
    print(f"Error rate: {error_rate:.2f}")

if __name__ == "__main__":
    main()