import numpy as np
import matplotlib.pyplot as plt
import math as math
import pandas as pd
from sklearn.datasets.samples_generator import make_classification
from mpl_toolkits.mplot3d import Axes3D

#建立逻辑回归的数据集（自己手动生成的数据集）
def create_data1():
    x = [[0, 0], [1, 0], [2, 1], [1.5, 1.5], [4.7, 2.1], [3.3, 2.9], [10.2, 11],
         [0, 1.1], [3, 5], [3.3, 4.5],
         [3, 7.2], [2, 15]]
    x = x+np.random.normal(size=(12, 2))*0.1
    y = [[1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0]]
    dataset = np.hstack((x, y))
    return dataset
#从sklearn库里面的make_classfication
def create_data2():
    X, labels = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                                    random_state=1, n_mius_per_class=1)
    dataset = np.hstack((X, np.mat(labels).T))
    return  dataset

#高斯分布的数据（利用协方差矩阵判断是否满足朴素贝叶斯）
def create_data3():
    mean1 = [0, 0]
    cov1 = [[1, 0.8], [0.8, 1]]
    data1 = np.random.miultivariate_normal(mean1, cov1, 100)
    mean2 = [3, 3]
    cov2 = [[1, 0.8], [0.8, 1]]
    data2 = np.random.miultivariate_normal(mean2, cov2, 100)
    zero = np.mat(np.zeros(100)).T
    data1 = np.hstack((data1, zero))
    one = np.mat(np.ones(100)).T
    data2 = np.hstack((data2, one))
    data =np.vstack((data1, data2))
    np.random.shuffle(data)
    return data


#从ucl数据集中得到数据
def get_ucldata():
    data =pd.read_csv("data_banknote_authentication.csv")
    data = data.values
    np.random.shuffle(data)
    #print(data)
    return data

def sigmoid(x):
    return 1/(1 + np.exp(-x))

#损失函数
def hloss(x, y, W):
    m = np.size(x, 0)
    loss = -np.dot(y.T, np.log(sigmoid(np.dot(x, W.T)))+1e-5) + np.dot((1 - y).T, np.log(1 - sigmoid(np.dot(x, W.T))+1e-5))
    return  loss/m


#利用梯度下降法求参数（regu参数true/false确定是否加正则项）
def gradien_desent(x,y,learning_rate,regu = False):
    x = np.column_stack((np.ones(np.size(x, 0)).T, x))
    n = np.size(x, 1)
    m = np.size(x, 0)
    W = np.mat(np.ones(n)).T
    old_loss = 0
    loss = hloss(x, y, W.T)
    loss_diff = loss-old_loss
    while(math.fabs(loss_diff)>1e-8):
        error = sigmoid(np.dot(x, W)) - y
        if(regu):
            W = W - learning_rate / m * np.dot(x.T, error)-learning_rate*W*0.5
        else:
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
#画图(二维)
def drawDomension2(W, data, string):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    x1 = np.linspace(-5, 10, 10000)
    x2 = (W[0, 0]+W[1, 0]*x1)/(-W[2, 0])
    for i in range(np.size(data,0)):
        if(data[i, 2]==0):
             plt.plot(data[i, 0], data[i, 1], "ro")
        else:
            plt.plot(data[i, 0].T, data[i, 1], "go")
    plt.plot(x1, x2)
    plt.title(string)
    plt.show()

#画图（三维）
def drawDimension3(W, data):
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
    colors1 = '#00CED1'  # 点的颜色
    colors2 = '#DC143C'
    ax.scatter(x1, y1, z1, c=colors1)
    ax.scatter(x2, y2, z2,)
    plt.show()


if __name__ == '__main__':
     data = create_data1()
     W = gradien_desent(data.T[np.arange(0, 2)].T, data.T[np.arange(2, 3)].T, 0.001)
     drawDomension2(W, data, "自己手动生成数据")
     print("自己手动生成数据的正确率(无正则化)")
     print(judge(np.column_stack((np.ones(np.size(data.T[np.arange(0, 2)].T, 0)).T,
                                  data.T[np.arange(0, 2)].T)), data.T[np.arange(2, 3)].T, W))
     W = gradien_desent(data.T[np.arange(0, 2)].T, data.T[np.arange(2, 3)].T, 0.001, True)
     drawDomension2(W, data, "自己手动生成数据")
     print("自己手动生成数据的正确率(正则化)")
     print(judge(np.column_stack((np.ones(np.size(data.T[np.arange(0, 2)].T, 0)).T,
                                  data.T[np.arange(0, 2)].T)), data.T[np.arange(2, 3)].T, W))
     data = get_ucldata()
     W = gradien_desent(data[0:1000, 0:4], data[0:1000, 4:5], 0.001)
     print("UCI上数据集的正确率")
     print(judge(np.column_stack((np.ones(np.size(data[1000:1300, 0:4], 0)).T, data[1000:1300, 0:4])), data[1000:1300, 4:5], W))
     drawDimension3(W, data)
     data = create_data2()
     W = gradien_desent(data.T[np.arange(0, 2)].T, data.T[np.arange(2, 3)].T, 0.001)
     drawDomension2(W, data, "利用sklearn库生成的数据")
     print("利用sklearn库生成的数据的正确率")
     print(judge(np.column_stack((np.ones(np.size(data.T[np.arange(0, 2)].T, 0)).T,
                             data.T[np.arange(0, 2)].T)), data.T[np.arange(2, 3)].T, W))
     data = create_data3()
     W = gradien_desent(data[0:80, 0:2], data[0:80, 2:3], 0.001)
     print("两个独立的高斯分布生成的数据的正确率")
     print(judge(np.column_stack((np.ones(np.size(data[80:100, 0:2], 0)).T, data[80:100, 0:2])), data[80:100, 2:3], W))
     drawDomension2(W, data, "两个独立的高斯分布")
