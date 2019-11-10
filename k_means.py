import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import  xlrd


#读取UCI中的一个聚类数据
def uci_data():
    data = pd.read_csv("seeds_dataset.csv")
    data = data.values
    return data[:, 0:7]

#利用两个高斯分布来看kmeans情况
def create_data_Guass():
    mean1 = [0, 0]
    cov1 = [[1, 0], [0, 1]]
    data1 = np.random.multivariate_normal(mean1, cov1, 100)
    mean2 = [5, 5]
    cov2 = [[1, 0], [0, 1]]
    data2 = np.random.multivariate_normal(mean2, cov2, 100)
    return  np.vstack((data1, data2))

#生成GMM的数据 3个高斯分布
def create_data_GMM():
    # 第一簇的数据
    num1, miu1, var1 = 400, [0, 3], [[1, 0], [0, 2]]
    X1 = np.random.multivariate_normal(miu1, var1, num1)
    # 第二簇的数据
    num2, miu2, var2 = 600, [5, 5], [[2, 0], [0, 3]]
    X2 = np.random.multivariate_normal(miu2, var2, num2)
    # 第三簇的数据
    num3, miu3, var3 = 1000, [-3, -3], [[3, 0], [0, 4]]
    X3 = np.random.multivariate_normal(miu3, var3, num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X

#更新z的后验概率
def update_γ(X, Miu, var, Pi):
    points, k = len(X), len(Pi)
    pdfs = np.zeros(((points, k)))
    for i in range(k):
        pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Miu[i], np.diag(var[i]))
    γ = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
    return  γ

def update_pi(γ):
    pi = γ.sum(axis=0)/γ.sum()
    return pi

#更新均值
def update_miu(X, γ, dimension):
    k =γ.shape[1]
    Miu = np.zeros((k, dimension))
    for i in range(k):
        Miu[i] = np.average(X, axis=0, weights=γ[:, i])
    return Miu

#更新方差
def update_var(X, miu, γ, dimension):
    k = γ.shape[1]
    var = np.zeros((k, dimension))
    for i in range(k):
        var[i] = np.average((X-miu[i])**2, axis=0, weights=γ[:, i])
    return var

#最大似然的log计算结果
def MLE(X, miu, var, pi):
    points = np.size(X, 0)
    k = np.size(pi, 0)
    pdfs = np.zeros(((points, k)))
    for i in range(k):
        pdfs[:, i] = pi[i] * multivariate_normal.pdf(X, miu[i], np.diag(var[i]))
    return np.mean(np.log(pdfs.sum(axis=1)))

#UCI数据的利用EM算法时的初始化
def initial_miu_var():
    miu = [[10, 16, 1, 6, 4, 4, 6],
           [12, 13, 0, 5, 3, 5, 5],
           [15, 14, 1, 5, 3, 3, 5]]
    var = [[1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1]]
    return miu, var


def EM(x, k, draw = True):
    #初始化
    k = k
    points = np.size(x, 0)
    dimension = np.size(x, 1)
    #我们生成数据时的高斯分布的均值和协方差
    standard_miu = [[0, 3], [5, 5],  [-3, -3]]
    standard_var = [[1, 2], [2, 3], [3, 4]]
    if(draw):
        miu = [[0, -1], [6, 0], [0, 9]]
        var = [[1, 1], [1, 1], [1, 1]]
    else:
        miu, var = initial_miu_var()
    pi = [1 / k] * 3
    γ = np.ones((points, k)) / k
    log = 0
    log_old = 10
    while(math.fabs(log-log_old)>1e-5):
        if(draw):
            draw_guass(X=x, standard_miu=standard_miu, standard_var=standard_var, miu=miu, var=var, pi= pi)
        log_old = log
        log = MLE(x, miu, var, pi)
        # print(log)
        #利用EM算法迭代更新
        γ = update_γ(x, miu, var, pi)
        miu = update_miu(x, γ,dimension)
        var = update_var(x, miu, γ, dimension)
        pi = update_pi(γ)
    return miu, var, pi

#x的概率密度函数
def px(x, pi, miu, var, k):
    sum = 0
    for i in range(k):
        sum = sum + pi[i]*multivariate_normal.pdf(x, miu[i], np.diag(var[i]))
    return sum

#GMM画图
def draw_guass(X, standard_miu, standard_var, miu=None, var=None, pi=None):
    colors = ['b', 'g', 'r']
    k = len(miu)
    n = len(X)
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -10, 10])
    max = 0
    index = 0
    for i in range(n):
        for j in range(k):
            γ_ij = pi[j]*multivariate_normal.pdf(X[i, :], miu[j], np.diag(var[j]))/px(x[i, :], pi, miu, var, k)
            if(γ_ij>max):
                max  = γ_ij
                index = j
        plt.plot(X[i, 0], X[i, 1], colors[index]+"o")
    ax = plt.gca()
    if (miu is not None) & (var is not None):
        for i in range(k):
            plot_args = {'fc': 'None', 'lw': 6, 'edgecolor': colors[i],  'ls': '-'}
            ellipse = Ellipse(miu[i], 3 * var[i][0], 3 * var[i][1], **plot_args)
            ax.add_patch(ellipse)
    plt.show()

#UCI数据的分类结果
def print_y(X, miu, var, pi):
    k = len(miu)
    n = len(X)
    max = 0
    index = 0
    for i in range(n):
        max = 0
        index = 0
        for j in range(k):
            γ_ij = pi[j] * multivariate_normal.pdf(X[i, :], miu[j], np.diag(var[j])) / px(x[i, :], pi, miu, var, k)
            if (γ_ij > max):
                max = γ_ij
                index = j
        print(index)

#更新每一类的中心
def refresh_center(center, x, y):
    n = np.size(x, 0)
    m = np.size(x, 1)
    k = np.size(center, 0)
    x_mean = np.zeros((k, m))
    for i in range(k):
        size = 0
        for j in range(n):
            if(y[j, 0]==i):
                size = size+1
                x_mean[i, :] = x_mean[i, :]+x[j, :]
        x_mean[i, :] = x_mean[i, :]/size
    return x_mean


def k_means(x, k):
    n = np.size(x, 0)
    #随机得到两个数的坐标，作为我们初始的中心点
    rand = random.sample(range(0, n), k)
    center = x[rand[0], :]
    for i in range(k-1):
        center = np.vstack((center, x[rand[i+1], :]))
    y = np.mat(np.zeros(n)).T
    old_y = np.mat(np.ones(n)).T
    while(not (old_y == y).all()):
        old_y = y.copy()
        for i in range(n):
            dis = 0
            min = 1000
            index = 0
            for j in range(k):
                dis = eruclidean_distance(center[j, :], x[i, :])
                if(dis<min):
                    min = dis
                    index = j
            y[i, 0] = index
        center = refresh_center(center, x, y)
    return x, y, center


def eruclidean_distance(x_center, x):
    dis = np.linalg.norm(x_center-x)
    return dis

#k_means的分类可视化
def draw(x ,y):
    n = np.size(x, 0)
    for i in range(n):
        if(y[i, 0]==0):
            plt.plot(x[i, 0], x[i, 1], "go")
        else:
            plt.plot(x[i, 0], x[i, 1], "ro")
    plt.show()

if __name__ == '__main__':
    x = create_data_Guass()
    x, y, cneter= k_means(x, 2)
    draw(x, y)
    x = uci_data()
    x, y, center = k_means(x, 3)
    print(y)
    x = uci_data()
    miu, var, pi = EM(x, 3, False)
    print_y(x, miu, var, pi)
    x = create_data_GMM()
    miu, var,pi = EM(x, 3)
    print(miu)
    print(var)
