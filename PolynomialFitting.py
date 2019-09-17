import numpy as np
import matplotlib.pyplot as plt
import math as math

#建立正弦函数数据集
def create_data(begin, end, size):
    X = np.arange(begin, end, (end-begin)/size)
    rand = (np.random.randn(1, 10)*0.1)
    Y = np.sin(2*math.pi*X)+rand
    mat_X = X.T
    mat_Y = Y.T
    data = np.column_stack((mat_X, mat_Y))
    return data

#建立多项式拟合中X矩阵
def create_x(m, x):
    data = np.ones(len(x)).T
    for i in range(1, m+1):
       data = np.column_stack((data, np.power(x, i)))
    return data

#解析法求解，无正则项
def normal_method(X,T):
     w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), T)
     return w

#解析法求解，带正则项
def normal_method_with_regular(X,T,lamda,m):
    w = np.dot(np.dot(np.linalg.inv(np.add(np.dot(X.T, X), lamda*np.eye(m+1))), X.T), T)
    return w

#梯度下降法
def gradient_descent(X, T, learning_rate,lamda):
     W = np.zeros(X.shape[1]).T
     gradient = np.dot(X.T, np.subtract(np.dot(X, W), T))+lamda*W
     for i in range(100000):
         W = np.subtract(W, gradient*learning_rate)
         gradient = np.dot(X.T, np.subtract(np.dot(X, W), T))+lamda*W
     return W

#共轭梯度法

#画图
def draw(w,data):
    p = np.poly1d(np.array(w.T)[::-1])
    x = np.linspace(0, 0.9, 1000)
    y = p(x)
    plt.plot(x, y)
    plt.plot(data[:, 0].T, data[:, 1].T,"ro")
    plt.show()


if __name__ == '__main__':
    data = create_data(0, 1, 10)
    X = create_x(9, data[:, 0])
    w = gradient_descent(X, data[:, 1],0.01,0.0001)
    draw(w, data)
