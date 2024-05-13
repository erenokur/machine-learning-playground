# probabilistic_training_classification

Probabilistic training classification is a method used to train a classification model by maximizing the posterior probability of the class labels given the input data. It's a common approach in machine learning for tasks like image recognition, spam filtering, and sentiment analysis.

The Probabilistic Classification algorithm works by estimating the probability of each class given the input features and then selecting the class with the highest probability as the predicted class. This method is based on Bayes' theorem, which provides a way to calculate the posterior probability of a class given the input data.

## Step-by-step Explanation of How the Algorithm Works

1. **Calculate the prior probability of each class based on the training data.**

   - Given a labeled training dataset consisting of input features `x` and corresponding class labels `C_k`, the algorithm learns the parameters of the probability distribution for each class `C_k` using the training data.
   - For each class `C_k`, the algorithm estimates the conditional probability distribution `p(x | C_k)`, which represents the likelihood of observing the input features `x` given that the true class is `C_k`.
   - Additionally, the algorithm estimates the prior probability `p(C_k)` for each class, representing the probability of encountering class `C_k` in the dataset.

2. **Estimate the likelihood of the input data given each class.**

   - Given a new input instance `x`, the algorithm computes the posterior probability `p(C_k | x)` for each class `C_k` using Bayes' theorem:
     ```
     P(C_k | x) = ( p(x | C_k) * P(C_k) ) / p(x)
     ```
   - The algorithm then selects the class `C_k` with the highest posterior probability as the predicted class:
     ```
     h(x) = argmax_k p(C_k | x)
     ```

3. **Use Bayes' theorem to calculate the posterior probability of each class given the input data.**

   - After making predictions on a test dataset, the algorithm's performance is evaluated using appropriate evaluation metrics such as accuracy, precision, recall, F1 score, or ROC curve analysis.

4. **Select the class with the highest posterior probability as the predicted class.**
   - Depending on the performance of the algorithm, it may undergo iterative refinement, which involves adjusting hyperparameters, feature selection, or model selection to improve classification accuracy.

### Step 1: Load and split the dataset

```
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
```

### Step 2: Train the Naive Bayes classifier

```
model = GaussianNB()
model.fit(X_train, y_train)
```

### Step 3: Predict probabilities for each class

```
predicted_probabilities = model.predict_proba(X_test)
```

### Step 4: Predict the class with the highest probability

```
predicted_classes = np.argmax(predicted_probabilities, axis=1)
```

### Step 5: Evaluate the model

```
accuracy = accuracy_score(y_test, predicted_classes)
print("Accuracy:", accuracy)
```

## Formulation

$$h(x) = argmax_k p(C_k | x) $$

Explanation of the terms:

    h(x): Prediction for the class Ck​ that maximizes the posterior probability.

    p(Ck​∣x): Posterior probability of class CkCk​ given input x.

    argmaxk​: Function that returns the index (class label) kk that maximizes the posterior probability.

### Installing

You need to have Python installed on your machine to run the projects.

```bash
pip install -r requirements.txt
```

### Recommended library versions

numpy==1.24.3
