import numpy as np
import matplotlib.pyplot as plt
import math as math
import pandas as pd

#建立逻辑回归的数据集（自己手动生成的数据集）
def create_data1():
    x = [[0, 0], [1, 0], [2, 1], [1.5, 1.5], [4.7, 2.1], [3.3, 2.9], [10.2, 11], [0, 1.1], [3, 5], [3.3, 4.5],
         [3, 7.2], [2, 15]]
    x = x+np.random.normal(size=(12, 2))*0.1
    y = [[1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0]]
    dataset = np.hstack((x, y))
    return dataset

#从ucl数据集中得到数据
def get_ucldata():
    data =pd.read_csv("transfusion.csv")
    data = data.values
    data = data[:, 1:5]
    return data

def sigmoid(x):
    return 1/(1 + np.exp(-x))

#损失函数，不带正则项
def hloss(x, y, β):
    m = np.size(x, 0)
    loss = -np.dot(y.T, np.log(sigmoid(np.dot(x, β.T)))+1e-5) + np.dot((1 - y).T, np.log(1 - sigmoid(np.dot(x, β.T))+1e-5))
    return  loss/m

#利用梯度下降法求参数
def gradien_desent(x,y,learning_rate):
    x = np.column_stack((np.ones(np.size(x, 0)).T, x))
    n = np.size(x, 1)
    m = np.size(x, 0)
   # print(x)
    β = np.mat(np.ones(n)).T
    old_loss = 0
    loss = hloss(x, y, β.T)
    loss_diff = loss-old_loss
    while(math.fabs(loss_diff)>2e-6):
        error = sigmoid(np.dot(x, β)) - y
        β = β - learning_rate/m*np.dot(x.T, error)
        old_loss = loss
        loss = hloss(x, y, β.T)
        loss_diff = loss-old_loss
    return β

#判断我们逻辑回归函数的正确性
def judge(x,y,β):
    m = np.size(x, 0)
    result = 0.0
    for i in range(m):
        if (sigmoid(np.dot(x[i],β))>0.5):
            temp_y = 1
            if(temp_y==y[i]):
                result = result+1
        else:
            temp_y = 0
            if (temp_y == y[i]):
                result = result + 1
    return result/m
#画图
def draw(β, data, string):
    x1 = np.linspace(0, 15, 10000)
    x2 = (β[0, 0]+β[1, 0]*x1)/(-β[2, 0])
    plt.plot(data[0:7, 0].T, data[0:7, 1].T, "ro")
    plt.plot(data[7:12, 0].T, data[7:12, 1].T, "go")
    plt.plot(x1, x2)
    plt.show()

if __name__ == '__main__':
     data = create_data1()
     β = gradien_desent(data.T[np.arange(0, 2)].T, data.T[np.arange(2, 3)].T, 0.01)
     draw(β, data, "hhh")
     print(judge(np.column_stack((np.ones(np.size(data.T[np.arange(0, 2)].T, 0)).T,
                                  data.T[np.arange(0, 2)].T)), data.T[np.arange(2, 3)].T, β))
     data = get_ucldata()
     β = gradien_desent(data[0:500, 0:3], data[0:500, 3:4], 0.001)
     print(judge(np.column_stack((np.ones(np.size(data[500:700, 0:3], 0)).T, data[500:700, 0:3])), data[500:700, 3:4], β))
