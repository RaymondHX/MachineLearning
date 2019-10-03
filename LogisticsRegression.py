import numpy as np
import matplotlib.pyplot as plt
import math as math

#建立逻辑回归的数据集（自己手动生成的数据集）
def create_data():
    x = [[0, 0], [1, 0], [2, 1], [1.5, 1.5], [4.7, 2.1], [3.3, 2.9], [10.2, 11], [0, 1.1], [3, 5], [3.3, 4.5],
         [6, 7.2], [13, 15]]
    x = x+np.random.normal(size=(12, 2))*0.1
    y = [[0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1]]
    dataset = np.hstack((x, y))
    return dataset

#损失函数，不带正则项
def hloss(x, y, β):
    m = np.size(x,0)
    loss = 0
    for i in range(m-1):
        loss =loss -y[i, 0]*math.log(1/(1+math.exp(-(np.dot(x[i],β.T)))), math.e)-\
              (1-y[i,0])*math.log(1-(1+math.exp(-(np.dot(x[i],β.T)))))
    return  loss

#利用梯度下降法求参数
def gradien_desent(x,y,learning_rate):
    m = np.size(x,0)
    x = np.hstack((np.arange(m).T, x))
    print(x)
    β = np.zeros(m)
    old_loss = 0
    loss = hloss(x, y, β)
    loss_diff = loss-old_loss
    while(loss_diff<1e-6):
        for j in range(m-1):
            for i in range(m-1):
                β[:, j] = β[:, j]-learning_rate*(1/m)*(1/(1+math.exp(-(np.dot(x[i],β.T))-y[i, 0])))*x[i, j]
        old_loss = loss
        loss = hloss(x, y, β)
        loss_diff = loss-old_loss


