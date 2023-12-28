import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

chunk_size = 2000  # Number of rows to read at a time
data_chunks = pd.read_csv('D:\\train.csv', chunksize=chunk_size)
first_chunk = next(data_chunks)
first_chunk.fillna(first_chunk.mean(), inplace=True)

# Split features and target variable
X = first_chunk.drop('target', axis=1)
y = first_chunk['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Create a scaler
scaler = StandardScaler()

# Fit the scaler on the feature training set from the first chunk
scaler.fit(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

model = SVC(kernel='linear')
for chunk in data_chunks:

    # Split features and target variable
    chunk.fillna(chunk.mean(), inplace=True)
    X = chunk.drop('target', axis=1)
    y = chunk['target']

    # Scale the features of each chunk
    X = scaler.transform(X)

    # Fit the SVR model
    model.fit(X, y)

test_score = model.score(X_test, y_test)
print("test score: {:.3f}".format(test_score))
print("Accuracy on testing set: {:.3f}".format(roc_auc_score(y_test, model.predict(X_test))))
print("classification report:\n", classification_report(y_test, model.predict(X_test), 
                                                        target_names=["non-5g", "5g"]))
