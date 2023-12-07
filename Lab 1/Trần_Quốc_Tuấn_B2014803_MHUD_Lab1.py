import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("baitap1.csv",delimiter=",") # Đọc file dữ liệu” “baitap1.csv”
print (data) # Hiển thị dữ liệu vừa đọc
print (data.iloc[:,2:3]) # Hiển thị tất cả dữ liệu cột số 3
print (data.loc[5:10]) # Hiển thị dữ liệu từ dòng 5 đến dòng 10
print (data.iloc[5:6,0:2]) # Hiển thị dữ liệu cột 1,2 của dòng 5
# Câu 6
data = sp.genfromtxt("baitap1.csv",delimiter=",")
x = data[:,1] # Lấy dữ liệu ở cột thứ 2 gắn cho biến X
y = data[:,2] # Lấy dữ liệu ở cột thứ 3 gắn cho biến Y
X = np.array(x) # Tạo mảng chứa X
Y = np.array(y) # Tạo mảng chứa y
plt.axis([0,100,0,100])
plt.plot(X,Y,"ro",color="blue")
plt.xlabel("Giá Trị Thuộc Tính X")
plt.ylabel("Giá Trị Thuộc Tính Y")
plt.show()
# Câu 7
for i in range(1,50):
    if i%2!=0:
        print(i)