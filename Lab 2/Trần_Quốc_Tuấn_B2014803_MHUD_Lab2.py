print("Cau A: Đọc dữ liệu từ tập dữ liệu.")
import pandas as pd
data = pd.read_csv("winequality-red.csv", sep=";")
print(data)

print("Cau B: ")
print("So luong phan tu", len(data))

import numpy as np
print("So luong nhan", len(np.unique(data.quality))) #6
print(data.quality.value_counts())
"""
5    681
6    638
7    199
4     53
8     18
3     10
Name: quality, dtype: int64
"""

print("Cau C: ")
X = data.iloc[:,0:11]
y = data.quality

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
print("So luong phan tu tap Test", len(X_test)) # 640
print("So luong nhan tap Test", len(np.unique(y_test))) # 6

print("Cau D: ")
print("D.i: ")
# Xây dựng mô hình K láng giềng KNN, với 7 láng giềng.
from sklearn.neighbors import KNeighborsClassifier
Mohinh_KNN = KNeighborsClassifier(n_neighbors = 7)
Mohinh_KNN.fit(X_train, y_train)

# Dự đoán nhãn cho các phần tử trong tập kiểm tra
y_pred = Mohinh_KNN.predict(X_test)

# Tính độ chính xác cho giá trị dự đoán của phần tử trong tập kiểm tra
from sklearn.metrics import accuracy_score
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)

# Tính độ chính xác cho giá trị dự đoán thông qua ma trận con
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred, labels=np.unique(data.quality).tolist()))
"""
Đáp án D.i
Accuracy is   50.31250000000001
[[  0   0   0   2   0   0]
 [  0   1  10  14   0   0]
 [  1   1 183  82   4   0]
 [  0   0 122 123  13   0]
 [  0   1  26  36  15   0]
 [  0   0   1   2   3   0]]
"""
print("D.ii: ")

y_pred_8 = Mohinh_KNN.predict(X_test.iloc[0:8,:])

print ("Accuracy is ", accuracy_score(y_test.iloc[0:8],y_pred_8)*100)
print(confusion_matrix(y_test.iloc[0:8], y_pred_8, labels=np.unique(data.quality).tolist()))

"""
Đáp án D.ii
Accuracy is  62.5
[[0 0 0 0 0 0]
 [0 0 0 0 0 0]
 [0 0 2 0 0 0]
 [0 0 2 3 1 0]
 [0 0 0 0 0 0]
 [0 0 0 0 0 0]]
"""

print("Cau E: ")
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
y_pred_Bayes = model.predict(X_test)

print ("Accuracy is ", accuracy_score(y_test,y_pred_Bayes)*100)
print(confusion_matrix(y_test, y_pred_Bayes, labels=np.unique(data.quality).tolist()))
"""
Đáp án: câu E
Accuracy is  54.84375
[[  1   1   0   0   0   0]
 [  1   1  13   9   1   0]
 [  1   9 181  70  10   0]
 [  0  10  69 127  46   6]
 [  0   0   6  29  41   2]
 [  0   0   0   2   4   0]]
"""

print("Cau F: ")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.0, random_state=1)
Mohinh_KNN = KNeighborsClassifier(n_neighbors = 7)
Mohinh_KNN.fit(X_train, y_train)
y_pred = Mohinh_KNN.predict(X_test)

accuracy_KNN = accuracy_score(y_test,y_pred)*100

model = GaussianNB()
model.fit(X_train, y_train)
y_pred_Bayes = model.predict(X_test)
accuracy_Bayes = accuracy_score(y_test,y_pred_Bayes)*100

print("accuracy_KNN", accuracy_KNN)
print("accuracy_Bayes", accuracy_Bayes)
"""
Đáp án: câu F
Accuracy_KNN 50.093808630394
Accuracy_Bayes 54.409005628517825
"""