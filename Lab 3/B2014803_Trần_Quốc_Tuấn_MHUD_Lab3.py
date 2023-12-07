# Câu 1: Xây dựng cây quyết định dựa vào chỉ số độ lợi thông tin và dự đoán nhãn

# Câu 1.a:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

wine_while_dt = pd.read_csv('winequality-white.csv', delimiter=';')

print(len(wine_while_dt))

# Câu 1.b:
col = wine_while_dt.shape[1]
print("Số lượng thuộc tính: ", col - 1)
print("Cột: '", wine_while_dt.iloc[:, -1].name, "' là nhãn")
print("Giá trị các nhãn: \n", np.unique(wine_while_dt.iloc[0:, -1]))

# Kết Quả:
'''
    Số lượng thuộc tính:  11
    Cột: ' quality ' là nhãn
    Giá trị các nhãn:
    [3 4 5 6 7 8 9]
'''

# 1.c:
kfd = KFold(n_splits=50, shuffle=True, random_state=42)

X = wine_while_dt.iloc[:, 0:11]
Y = wine_while_dt.quality

gini = DecisionTreeClassifier(
    criterion="entropy", random_state=100, max_depth=8, min_samples_leaf=5)

train_index, test_index = next(kfd.split(X))
X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
print("Tập dữ liẹu test có", len(X_test), 'phần tử')
print("Tập dữ liẹu train có", len(X_train), 'phần tử')
i = 0

# Kết Quả:
'''
    Tập dữ liẹu test có 98 phần tử
    Tập dữ liẹu train có 4800 phần tử
'''
# 1.(d, e, f, g):

total_acc = 0
KNN = KNeighborsClassifier(n_neighbors=9)
model = GaussianNB()

for train_idex, test_index in kfd.split(X):
    X_train, X_test = X.iloc[train_index, ], X.iloc[test_index, ]
    y_train, Y_test = Y.iloc[train_index, ], Y.iloc[test_index, ]
    i += 1
    gini.fit(X_train, y_train)
    KNN.fit(X_train, y_train)
    model.fit(X_train, y_train)
    y_pred = gini.predict(X_test)
    y_pred_knn = KNN.predict(X_test)
    print('-----------')
    print("Độ chính xác cho phần tử lớp ", i, "KNN là: ",
          round(accuracy_score(Y_test, y_pred_knn) * 100), '%')
    dubao = model.predict(X_test)
    print("Độ chính xác cho phần tử lớp ", i, "bayes là: ",
          round(accuracy_score(Y_test, dubao) * 100), '%')
    lb = np.unique(y_pred)
    total_acc += accuracy_score(Y_test, y_pred)
    print("Độ chính xác cho phần tử lớp ", i, "là: ",
          round(accuracy_score(Y_test, y_pred) * 100), '%')
    print('-----------')

print("Độ chính xác tổng thể là: ", round(total_acc * 100 / 50), '%')

# Kết Quả:
'''
    -----------
    Độ chính xác cho phần tử lớp  1 KNN là:  46 %
    Độ chính xác cho phần tử lớp  1 bayes là:  54 %
    Độ chính xác cho phần tử lớp  1 là:  55 %
    -----------
    -----------
    Độ chính xác cho phần tử lớp  2 KNN là:  60 %
    Độ chính xác cho phần tử lớp  2 bayes là:  36 %
    Độ chính xác cho phần tử lớp  2 là:  64 %
    -----------
    -----------
        ...
        ...
    -----------
    -----------
    Độ chính xác cho phần tử lớp  49 KNN là:  62 %
    Độ chính xác cho phần tử lớp  49 bayes là:  49 %
    Độ chính xác cho phần tử lớp  49 là:  68 %
    -----------
    -----------
    Độ chính xác cho phần tử lớp  50 KNN là:  48 %
    Độ chính xác cho phần tử lớp  50 bayes là:  45 %
    Độ chính xác cho phần tử lớp  50 là:  61 %
    -----------
    Độ chính xác tổng thể là:  64 %
'''

# Câu 2:
X = np.array([[180, 15, 0],
             [167, 42, 1],
             [136, 35, 1],
             [174, 15, 0],
             [141, 28, 1]])

labels = [0, 1, 1, 0, 1]

cif_gini = DecisionTreeClassifier(
    criterion="gini", random_state=10, min_samples_leaf=2)
cif_gini.fit(X, labels)
result = cif_gini.predict([[135, 39, 1]])
print("Người này là: ", result)

# Kết Quả:
'''
    Người này là:  [1]
'''
