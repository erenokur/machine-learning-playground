# bayesian_decision_tree_classifier

the Bayesian Decision Tree classifier combines the concepts of Bayesian inference and decision trees to provide a probabilistic framework for classification tasks, allowing for uncertainty quantification and robustness to noisy data. This classifier is based on the idea of learning a decision tree structure that maximizes the posterior probability of the class labels given the input data. By incorporating Bayesian inference, the model can handle uncertainty in the data and make more informed decisions about the class labels.

## Formulation

The Bayesian Decision Tree (BDT) classifier combines the principles of Bayesian inference with decision trees to provide a probabilistic framework for classification tasks.

### Bayesian Inference

Bayes' theorem is central to Bayesian inference and provides a mathematical framework for updating our beliefs about the parameters of a model based on observed data. Given a set of data D and a set of model parameters θ.

$$ P(θ∣D)= \frac{P(D∣θ)⋅P(θ)​}{P(D)} $$

The Bayesian formula consists of the following components:

- **Posterior Probability (P(θ∣D))**: This is the probability of the parameters given the data. It's what we want to compute using the Bayesian formula.

- **Likelihood (P(D∣θ))**: This is the probability of the data given the parameters. It's typically easy to compute.

- **Prior Probability (P(θ))**: This is the initial or existing probability of the parameters before seeing the data. It represents our prior belief about the parameters.

- **Marginal Likelihood (P(D))**: This is the total probability of the data. It's used to normalize the Bayesian formula to make sure the total probability adds up to 1.

### Bayesian Decision Trees

Bayesian Decision Tree, on the other hand, is a probabilistic extension of the traditional decision tree algorithm that incorporates Bayesian inference to handle uncertainty in the data and make more informed decisions about the class labels.

$$ P(y_i | x_i,y_p) = \frac{P(x_i | y_i,y_p)P(y_i|y_p)}{P(x_i|y_p)} $$

- **Likelihood $P(x_i​∣y_i​,y_p​)$**:

        The likelihood represents the probability of observing a feature vector x_i​ given specific class labels y_i​ and its parent node's class label y_p​.

        In the context of a decision tree node, this likelihood helps quantify how well the feature vector x_i​ fits the predicted class labels y_i​ and y_p​.

        It contributes to the calculation of the posterior probability of class labels at each decision point.

- **Prior Probability $P(y_i∣y_p)$** :

        The prior probability represents the likelihood of a particular class label yiyi​ given its parent node's class label ypyp​ without considering any observed features.

        In the context of a decision tree node, this prior probability reflects our belief about the distribution of class labels at that node before observing any data.

         It serves as an initial estimate of the probability of class labels and influences the posterior probability calculation.

- **Marginal Likelihood $P(x_i∣y_p)$**:

        The marginal likelihood represents the probability of observing the feature vector x_i​ given only the parent node's class label y_p​, without considering the specific class label y_i​ at the current node.

        In the context of a decision tree node, this marginal likelihood helps capture the overall distribution of features given the parent node's class label.

        It normalizes the likelihood term to ensure that the posterior probability calculation accounts for the entire feature space.

## Step-by-step Explanation of How the Algorithm Works

### Importing Libraries

We import necessary libraries including NumPy for numerical computations, Counter from collections for counting occurrences, and functions from scikit-learn (load_iris, train_test_split, accuracy_score) for loading the Iris dataset, splitting it into train and test sets, and calculating accuracy, respectively.

### BayesianDecisionTree Class Definition

The `BayesianDecisionTree` class is defined, which serves as the implementation of the Bayesian Decision Tree classifier. It includes the following methods:

- **Constructor (`__init__`)**: Initializes the parameters `max_depth` and `min_samples_split` which control the maximum depth of the tree and the minimum number of samples required to split an internal node, respectively.

- **`fit` method**: Trains the classifier by growing the decision tree using the `_grow_tree` method.

- **`_grow_tree` method**: Recursively builds the decision tree by selecting the best feature and split point at each node. It terminates when the maximum depth is reached, or when the number of samples is too small, or when all samples belong to the same class.

- **`_bayesian_score` method**: Computes the Bayesian posterior probability of a node based on the class label counts.

- **`predict` method**: Predicts the class labels for a set of input samples using the decision tree.

- **`_predict_tree` method**: Recursively predicts the class label for a single input sample based on the decision tree.

### Loading and Preparing Data

The Iris dataset is loaded using `load_iris()` and split into training and test sets using `train_test_split()`.

### Training the Classifier

An instance of the `BayesianDecisionTree` classifier (`bdt_classifier`) is created and trained using the training data (`X_train, y_train`) with the `fit` method.

### Making Predictions

Predictions are made on the test dataset (`X_test`) using the `predict` method of the trained classifier (`bdt_classifier`).

### Calculating Accuracy

The accuracy of the classifier is calculated by comparing the predicted labels with the true labels from the test dataset using `accuracy_score` from scikit-learn.

### Printing the Accuracy

The accuracy of the Bayesian Decision Tree classifier is printed to the console.
