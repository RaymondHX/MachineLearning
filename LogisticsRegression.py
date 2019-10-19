import numpy as np
import matplotlib.pyplot as plt
import math as math
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

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
    data =pd.read_csv("data_banknote_authentication.csv")
    data = data.values
    np.random.shuffle(data)
    print(data)
    return data

def sigmoid(x):
    return 1/(1 + np.exp(-x))

#损失函数，不带正则项
def hloss(x, y, W):
    m = np.size(x, 0)
    loss = -np.dot(y.T, np.log(sigmoid(np.dot(x, W.T)))+1e-5) + np.dot((1 - y).T, np.log(1 - sigmoid(np.dot(x, W.T))+1e-5))
    return  loss/m

#利用梯度下降法求参数
def gradien_desent(x,y,learning_rate):
    x = np.column_stack((np.ones(np.size(x, 0)).T, x))
    n = np.size(x, 1)
    m = np.size(x, 0)
    W = np.mat(np.ones(n)).T
    old_loss = 0
    loss = hloss(x, y, W.T)
    loss_diff = loss-old_loss
    while(math.fabs(loss_diff)>2e-8):
        error = sigmoid(np.dot(x, W)) - y
        W = W - learning_rate/m*np.dot(x.T, error)
        old_loss = loss
        loss = hloss(x, y, W.T)
        loss_diff = loss-old_loss
    return W

#判断我们逻辑回归函数的正确性
def judge(x,y,W):
    m = np.size(x, 0)
    result = 0.0
    for i in range(m):
        if (sigmoid(np.dot(x[i],W))>0.5):
            if(y[i]==1):
                result = result+1
        else:
            if (y[i]==0):
                result = result + 1
    return result/m
#画图
def drawDomension2(W, data, string):
    x1 = np.linspace(0, 15, 10000)
    x2 = (W[0, 0]+W[1, 0]*x1)/(-W[2, 0])
    plt.plot(data[0:7, 0].T, data[0:7, 1].T, "ro")
    plt.plot(data[7:12, 0].T, data[7:12, 1].T, "go")
    plt.plot(x1, x2)
    plt.show()

def drawDimension3(W, data):
    print(np.shape(data))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1 = []
    y1 = []
    z1 = []
    x2 = []
    y2 = []
    z2 = []
    for i in range(np.size(data, 0)):
        if(data[i, 3]==0):
            x1.append(data[i, 0])
            y1.append(data[i, 1])
            z1.append(data[i, 2])
        else:
            x2.append(data[i, 0])
            y2.append(data[i, 1])
            z2.append(data[i, 2])
    ax.scatter(x1, y1, z1)
    ax.scatter(x2, y2, z2)
    plt.show()


if __name__ == '__main__':
     data = create_data1()
     W = gradien_desent(data.T[np.arange(0, 2)].T, data.T[np.arange(2, 3)].T, 0.001)
     drawDomension2(W, data, "hhh")
     print(judge(np.column_stack((np.ones(np.size(data.T[np.arange(0, 2)].T, 0)).T,
                                  data.T[np.arange(0, 2)].T)), data.T[np.arange(2, 3)].T, W))
     data = get_ucldata()
     W = gradien_desent(data[0:1000, 0:4], data[0:1000, 4:5], 0.001)
     print(judge(np.column_stack((np.ones(np.size(data[1000:1300, 0:4], 0)).T, data[1000:1300, 0:4])), data[1000:1300, 4:5], W))
     drawDimension3(W, data)


