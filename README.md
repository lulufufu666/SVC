
# Project Description

This project uses a tumor dataset with missing values and applies Support Vector Machine (SVM) for classification to predict whether a tumor is malignant.

# Project Files

File | Description
--- | ---
train.csv | Tumor dataset
tumor_prediction.py | Code for tumor prediction using SVM
README.md | Project documentation

# Dependencies

1. pandas
2. scikit-learn

# Code Explanation

1. Reading the file

The data set is read using the `read_csv` function from pandas.

2. Data preprocessing

The data set is read in chunks of 2000 rows at a time and missing values are filled with the mean.

3. Splitting features and target variable

The data set is divided into feature variables `X` and target variable `y`.

4. Splitting the data

The data is split into training and testing sets using the `train_test_split` function with a ratio of 70:30.

5. Creating a scaler

A `StandardScaler` object is created to standardize the features.

6. Scaling the training set

The training set is standardized using the scaler.

7. Model training

The training set is used to train a Support Vector Machine (SVM) model.

8. Testing set evaluation

The model's performance is evaluated on the testing set by computing accuracy and ROC AUC (Area Under the Curve).

# How to Run

1. Make sure the dependencies are installed.
2. Save the code as `tumor_prediction.py`.
3. Save the dataset as `train.csv`.
4. Run the code by executing the command `python tumor_prediction.py` in the console.

# Conclusion

Our model achieves an accuracy of `test score: 0.800` and an AUC of `Accuracy on testing set: 0.600` on the testing set, indicating that our SVM performs moderately in predicting whether a tumor is malignant. Further optimization of the model or trying other algorithms can be explored based on specific requirements.
