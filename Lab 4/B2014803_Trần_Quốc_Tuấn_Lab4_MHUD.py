'''Bai 1'''
print("Câu 1:")
import numpy as np
import pandas as pd
dt = pd.read_csv("Housing_2019.csv", index_col=0)
X=dt.iloc[:,[1,2,4,10]]
Y = dt.price

import sklearn 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1.0/3, random_state=42)
len(X_train)

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(random_state=0)

bagging_regtree = BaggingRegressor(estimator=tree, n_estimators = 10,  random_state= 42)
bagging_regtree.fit(X_train, y_train)
y_pred = bagging_regtree.predict(X_test)

from sklearn.metrics import mean_squared_error
err= mean_squared_error(y_test, y_pred)
print("MSE LinearRegression = ", err)
rmse_err = np.sqrt(err)
print("RMSE LinearRegression = ", round(rmse_err,3))

'''Bai 2'''
print("Câu 2:")
from sklearn.datasets import load_wine
dt = load_wine()
dt.data[2:14]
dt.target[1:2,]

import sklearn 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dt.data, dt.target, test_size=1.0/3, random_state=42)
len(X_train)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=3, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error
err =mean_squared_error(y_test, y_pred)
print("MSE LinearRegression = ", err)
rmse_err = np.sqrt(err)
print("RMSE LinearRegression = ", round(rmse_err,3))

'''Bai 3'''
print("Câu 3:")
from sklearn.datasets import load_wine
dt = load_wine()
X=dt.data
Y=dt.target

from sklearn.model_selection import KFold
kf=KFold(n_splits=3)

for train_index, test_index in kf.split(X):
    #print("Train: ", train_index, "Test: ", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    print("X_test: ", len(X_test))
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(max_depth=3, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    from sklearn.metrics import mean_squared_error
    err =mean_squared_error(y_test, y_pred)
    print("MSE LinearRegression = ", err)
    rmse_err = np.sqrt(err)
    print("RMSE LinearRegression = ", round(rmse_err,3))