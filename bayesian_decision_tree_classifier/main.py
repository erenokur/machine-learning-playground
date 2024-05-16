import numpy as np
from collections import Counter

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class BayesianDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Constructor for Bayesian Decision Tree classifier.
        args:
            max_depth: int, optional
                The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure.
            min_samples_split: int, optional
                The minimum number of samples required to split an internal node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively builds a decision tree by selecting the best feature and split point at each node.
        args:
            X: numpy array
                The input data of shape (n_samples, n_features).
            y: numpy array
                The target values of shape (n_samples,).
            depth: int
                The current depth of the tree.
        returns:
            tuple or int
                A tuple (best_feature, best_split_point, left_subtree, right_subtree) if the node is split,
                otherwise an integer representing the class label.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check termination conditions
        if (
            depth == self.max_depth
            or n_samples < self.min_samples_split
            or n_labels == 1
        ):
            return Counter(y).most_common(1)[0][0]

        # Select best feature and split point
        best_feature, best_split_point = None, None
        best_score = -1
        for feature in range(n_features):
            unique_values = np.unique(X[:, feature])
            for value in unique_values:
                left_indices = np.where(X[:, feature] <= value)[0]
                right_indices = np.where(X[:, feature] > value)[0]
                
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                # Compute score (Bayesian posterior probability)
                left_label_counts = Counter(y[left_indices])
                right_label_counts = Counter(y[right_indices])
                left_score = self._bayesian_score(left_label_counts)
                right_score = self._bayesian_score(right_label_counts)
                score = (left_score * len(left_indices) + right_score * len(right_indices)) / n_samples

                if score > best_score:
                    best_score = score
                    best_feature = feature
                    best_split_point = value

        if best_score == -1:
            return Counter(y).most_common(1)[0][0]

        left_indices = np.where(X[:, best_feature] <= best_split_point)[0]
        right_indices = np.where(X[:, best_feature] > best_split_point)[0]

        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_split_point, left_subtree, right_subtree)

    def _bayesian_score(self, label_counts):
        """
        Computes the Bayesian posterior probability of a node.
        args:
            label_counts: dict
                A dictionary containing the counts of each class label in the node.
        returns:
            float
                The Bayesian posterior probability of the node.
        """
        total_samples = sum(label_counts.values())
        score = 0
        for count in label_counts.values():
            p = count / total_samples
            score -= p * np.log(p)
        return score

    def predict(self, X):
        """
        Predicts the class labels for a set of input samples.
        args:
            X: numpy array
                The input data of shape (n_samples, n_features).
        returns:
            numpy array
                The predicted class labels of shape (n_samples,).
        """
        predictions = np.array([self._predict_tree(x, self.tree) for x in X])
        return predictions

    def _predict_tree(self, x, tree):
        """
        Recursively predicts the class label for a single input sample.
        args:
            x: numpy array
                The input sample of shape (n_features,).
            tree: tuple or int
                The decision tree node.
        returns:
            int
                The predicted class label.
        """
        if not isinstance(tree, tuple):
            return tree
        feature, split_point, left_subtree, right_subtree = tree
        if x[feature] <= split_point:
            return self._predict_tree(x, left_subtree)
        else:
            return self._predict_tree(x, right_subtree)


# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build and train Bayesian Decision Tree classifier
bdt_classifier = BayesianDecisionTree()
bdt_classifier.fit(X_train, y_train)

# Make predictions
predictions = bdt_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Bayesian Decision Tree Classifier Accuracy:", accuracy)
