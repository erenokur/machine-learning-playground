# Titanic: Machine Learning from Disaster

This is a project to predict the survival of passengers on the Titanic using machine learning models. The dataset is from the Kaggle competition [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic).

## Getting Started

The goal is to predict whether passengers survived or did not survive the Titanic disaster using machine learning models.

### Approach and Rationale:

1. Data Loading and Preparation:
   - Loading Data: The data is read from CSV files into pandas DataFrames. This allows for easy manipulation and analysis.
   - Setting Up Folders: The input and output folders are defined to manage file paths and save results systematically.
2. Data Exploration and Visualization:
   - Understanding Relationships: Pivot tables and visualizations (e.g., histograms, point plots) are created to understand the relationships between features (e.g., Pclass, Sex, Age) and the target variable (Survived).
   - Rationale: Visualization helps in identifying important features and potential patterns that can improve model accuracy.
3. Data Cleaning and Feature Engineering:
   - Dropping Irrelevant Features: Features like Ticket and Cabin are dropped because they may not provide significant predictive power or are too sparse.
   - Extracting Titles: Titles are extracted from the Name feature to create a new feature, which might correlate with social status and survival likelihood.
   - Handling Missing Values: Missing values, especially in Age, are handled by estimating values based on other features like Pclass and Sex.
   - Categorical to Numerical Conversion: Converting categorical features (e.g., Sex, Embarked, Title) to numerical values to make them usable in machine learning models.
   - Creating New Features: New features like FamilySize and IsAlone are created to capture additional information about passengers' family structures.
4. Data Transformation:
   - Age Binning: Ages are grouped into bins to reduce the impact of outliers and capture age groups' effect on survival.
   - Fare Binning: Similar to Age, Fare is also categorized to handle skewness and make the data more manageable for algorithms.
5. Model Training and Evaluation:
   - Training Various Models: Multiple machine learning models (e.g., Logistic Regression, SVM, KNN, Random Forest, Naive Bayes, Decision Tree) are trained to predict survival.
   - Rationale: Using a variety of models allows comparison and selection of the best-performing model for this specific problem.
   - Model Evaluation: Accuracy scores are calculated for each model to evaluate and compare their performance on the training data.
6. Model Selection and Submission:
   - Selecting Best Model: The model with the highest accuracy is chosen for final predictions.
   - Generating Predictions: The selected model is used to predict survival on the test dataset.
   - Submission: The predictions are saved in the required format for submission to the Kaggle competition.

## Why These Steps Are Taken

1. **Data Exploration**: Helps in understanding the dataset and identifying key features.
2. **Data Cleaning**: Removes irrelevant or redundant features and handles missing values.
3. **Feature Engineering**: Creates new features or transforms existing ones to improve model performance.
4. **Data Transformation**: Prepares the data for machine learning algorithms by converting categorical features to numerical values and binning continuous features.
5. **Model Training**: Trains multiple models to compare their performance and select the best one.
6. **Model Evaluation**: Evaluates models based on accuracy scores to choose the most accurate model for predictions.

## Recommended library versions

pandas==2.0.3

numpy==1.24.3

scipy==1.10.1

seaborn==0.13.2

matplotlib==3.7.5
