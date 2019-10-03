import numpy as np
import matplotlib.pyplot as plt
import math as math

#建立正弦函数数据集
#生成从begin到end大小为size的数据集
def create_data(begin, end, size):
    X = np.arange(begin, end, (end-begin)/size)
    rand = (np.random.randn(1, size)*0.1)
    Y = np.sin(2*math.pi*X)+rand
    data = np.column_stack((X.T, Y.T))
    return data

#建立多项式拟合Xw = T 这里的X矩阵
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
     W = np.zeros(10).T
     #初始化一个W矩阵，这里10是因为当前为9次多项式，一共有10个参数
     gradient = np.dot(X.T, np.subtract(np.dot(X, W), T))+lamda*W
     while np.all(np.absolute(gradient)>1e-9):
         W = np.subtract(W, gradient*learning_rate)
         gradient = np.dot(X.T, np.subtract(np.dot(X, W), T))+lamda*W
     return W

#共轭梯度法
def conjugate_gradient(X, T, lamda):
    W = np.zeros(10).T
    Q = np.dot(X.T, X) + lamda * W
    r = -(np.dot(X.T, np.subtract(np.dot(X, W), T))+lamda*W)
    p = r
    for k in range(X.shape[0]+1):
        α = np.dot(r.T, r)/np.dot(np.dot(p.T, Q), p)
        W = np.add(W, np.dot(α, p))
        old_r = r
        r = r - np.dot(np.dot(α, Q), p)
        β = np.dot(r.T, r)/np.dot(old_r.T, old_r)
        p = r +np.dot(β, p)
        k = k+1
    return W

#画图
def draw(w, data, string):
    #p为我们的拟合多项式
    p = np.poly1d(np.array(w.T)[::-1])
    x = np.linspace(0, 0.9, 1000)
    y = p(x)
    plt.plot(x, y)
    #画出数据集中的点
    plt.plot(data[:, 0].T, data[:, 1].T, "ro")
    plt.title(string)
    plt.show()


if __name__ == '__main__':
    data = create_data(0, 1, 10)
    X = create_x(9, data[:, 0])
    #这里第一个参数 9 代表用9次多项式去拟合
    w = conjugate_gradient(X, data[:, 1], 0.00001)
    draw(w, data, "conjugate gradient")
    w = normal_method(X,data[:, 1])
    draw(w, data, "normal_method")
    w = normal_method_with_regular(X, data[:, 1], 0.0001, 9)
    draw(w, data, "normal_method_with_regular")
    w = gradient_descent(X, data[:, 1], 0.001, 0.000001)
    draw(w, data, "gradient descent")