# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Dataset
df = pd.read_csv('data.csv')
# : required to get a matrix(X), and should not be used for a vector(y)
X = df.iloc[:, :-1].values
y = df.iloc[:, 5].values

# Pre-Processing -------------------------------------------------------------------------
# Missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, :-1])
X[:, :-1] = imputer.transform(X[:, :-1])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#Avoid Dummy Variable Trap
X = X[:, 1:]

# Splitting to train and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_X = sc_X.fit(X_train)
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)

# Regression ------------------------------------------------------------------------------
# Simple Regressor fitting
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting
y_pred = regressor.predict(X_test)

# Visualizing
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('TITLE')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Multiple linear Regression
# Same as Simple Linear Regression

# Backward Elimination
# Prepending constant 1 column for backward elimination
import statsmodel.formula.api as sm
X = np.append(arr = np.ones((no_of_cols, 1)).astype(int), values = X, axis = 1)
# Repeat by deleting (one highest)column in X_opt whose P value in summary is > predefined threshold (5%)
X_opt = X[: [0, 1, 2, 3, 4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Polynomial Linear Regression

