import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, svm, metrics

training_data = pd.read_csv('train.csv')

y = np.array(list(training_data['SalePrice']))
X = training_data.drop(columns=['SalePrice', 'Id'])

print(X.columns)

# Numeric 
mappings = {}
numeric_X = pd.DataFrame()

def get_encoding(col, v):
    '''
        Convert strings into encodings. Tracks encodings with global
        variables mappings.

        Params:
            col (str): column name
            v (str or int): value from column

        Returns:
            (int): integer value or numeric encoding for string
    '''
    if type(v) == int or type(v) == float:
        if np.isnan(v):
            return 0
        return int(v)
    if not col in mappings:
        mappings[col] = [v]
        return 0
    if not v in mappings[col]:
        mappings[col].append(v)
        return len(mappings[col]) - 1
    return mappings[col].index(v)

for column in X.columns:
    num_col = [get_encoding(column, x) for x in X[column]]
    numeric_X[column] = num_col
scaler = preprocessing.StandardScaler().fit(numeric_X)
numeric_X = scaler.transform(numeric_X)

prescale = False
y = y.reshape(-1, 1)
y_scaler = preprocessing.StandardScaler().fit(y)
y = y_scaler.transform(y)

#pd.DataFrame(numeric_X).to_csv('test_encoding.csv', index=False)
X_train, X_val, y_train, y_val = model_selection.train_test_split(numeric_X, y, test_size=0.2)

# SVM Classifier
svm_classifier = svm.SVR(kernel='poly', degree=3, max_iter=5000)
svm_classifier.fit(X_train, y_train)
y_val_pred = svm_classifier.predict(X_val)

predictions = pd.DataFrame()
predictions['prediction'] = y_val_pred
predictions['y_val'] = y_val
predictions.to_csv("validation_results.csv", index=False)

rmse_train = metrics.mean_squared_error(y_train, svm_classifier.predict(X_train))
print("SVR Training RMSE: %f" % rmse_train)
rmse_val = metrics.mean_squared_error(y_val, y_val_pred)
print("SVR Validation RMSE: %f" % rmse_val)

# Generate test predictions and submission file
test_data = pd.read_csv('test.csv')
ids = test_data['Id']
X_test = test_data.drop(columns=['Id'])

numeric_X_test = pd.DataFrame()
for column in X_test.columns:
    num_col = [get_encoding(column, x) for x in X_test[column]]
    numeric_X_test[column] = num_col
numeric_X_test = scaler.transform(numeric_X_test)

y_test_scaled = svm_classifier.predict(numeric_X_test)
y_test = y_scaler.inverse_transform(y_test_scaled.reshape(-1,1))

output = pd.DataFrame()
output.insert(0, 'Id', ids)
print(y_test)
output.insert(1, 'SalePrice', y_test)
output.to_csv("submission.csv", index=False)