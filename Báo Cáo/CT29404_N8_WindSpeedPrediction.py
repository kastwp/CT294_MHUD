import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('wind_dataset.csv')
print("--------------------\n")

print(" Kiểm tra null")
print(df.isnull().sum())
print("--------------------\n")

print(" Các giá trị có tần suất xuất hiện nhiều:")
colu = ["IND.1","T.MAX" ,"IND.2" ,"T.MIN" ,"T.MIN.G" ]
for i in colu:
    print(i)
    print(df[i].mode())
ModeInd1 = 0.0
ModeTmax = 10.0
ModeInd2 = 0.0
ModeTMIN = 9.0
ModeTMIN_g = 5.0
df["IND.1"].fillna(ModeInd1 , inplace = True)
df["T.MAX"].fillna(ModeTmax , inplace = True)
df["IND.2"].fillna(ModeInd2 , inplace = True)
df["T.MIN"].fillna(ModeTMIN , inplace = True)
df["T.MIN.G"].fillna(ModeTMIN_g , inplace = True)
print("--------------------\n")

print(" Kiểm tra null")
print(df.isnull().sum())
print("--------------------\n")

sns.boxplot(x=df['WIND'])
#plt.show()
for i in np.where(df["WIND"]>=23):
    df.drop(i,inplace = True)
sns.boxplot(x=df['RAIN'])
#plt.show()
for k in np.where(df["RAIN"]>=25):
    df.drop(k,inplace =True)

print(df)
print("--------------------\n")

from sklearn.model_selection import train_test_split
X = df.iloc[:, 2:9]
y = df['WIND']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.0, random_state=5)

print(" KNN")
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
knn = KNeighborsRegressor(n_neighbors=9)
knn.fit(X_train, y_train)
predict_knn = knn.predict(X_test)
print('MAE: ', metrics.mean_absolute_error(y_test, predict_knn))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predict_knn)))
print("--------------------\n")

print(" Linear regression")
from sklearn.linear_model import LinearRegression
from sklearn import metrics
reg = LinearRegression()
reg.fit(X_train,y_train)
predict_reg = reg.predict(X_test)
print('MAE: ', metrics.mean_absolute_error(y_test, predict_reg))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predict_reg)))
print("--------------------\n")

print(" Decision Tree")
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
tree = DecisionTreeRegressor()
tree.fit(X_train,y_train)
predict_tree = tree.predict(X_test)
print('MAE: ' ,metrics.mean_absolute_error(y_test, predict_tree))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predict_tree)))
print("--------------------\n")


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv("wind_dataset.csv", sep=",")

accuracies = []
A = df.iloc[:, 2:9]
b = df['WIND']
for i in range(10):
    A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=1/3.0, random_state=42 + i)
    model = DecisionTreeRegressor(max_depth=21, random_state=42, min_samples_leaf=14)
    model.fit(A_train, b_train)
    b_pred = model.predict(A_test)
    accuracy = mean_absolute_error(b_test, b_pred)
    print("--------------------\n")
    print("Độ chính xác ở phân lớp:", i + 1)
    print("Max_depth:", 12 + i)
    print("Min_samples_leaf:", 5 + i)
    print("MAE DecisionTree: ", accuracy)
    knn = KNeighborsRegressor(n_neighbors=9)
    knn.fit(A_train, b_train)
    predict_knn = knn.predict(A_test)
    print('MAE KNN: ', metrics.mean_absolute_error(b_test, predict_knn))
    reg = LinearRegression()
    reg.fit(A_train, b_train)
    predict_reg = reg.predict(A_test)
    print('MAE Linear regression: ', metrics.mean_absolute_error(b_test, predict_reg))
