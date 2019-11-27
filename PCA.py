import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
def create_data():
    s1 = np.mat(np.random.normal(2, 3, 20)).T
    s2 = np.mat(np.random.normal(4, 4, 20)).T
    s3 = np.mat(np.random.normal(0, 0.5, 20)).T
    s = np.hstack((s1, s2))
    s = np.hstack((s, s3))
    return s

def loadImage():
    img = Image.open("testImg.jpg")
    img = img.convert("L")
    width = img.size[0]
    height = img.size[1]
    data = img.getdata()
    data = np.array(data).reshape(height, width) / 100
    # 查看原图的话，需要还原数据
    new_im = Image.fromarray(data * 100)
    new_im.show()
    return data

def loadImages():

    for i in range(10):
        img = Image.open("test"+str(i)+".jpg")
        img = img.convert("L")
        width = img.size[0]
        height = img.size[1]
        data = img.getdata()
        data = np.array(data).reshape(height*width)
        if(i == 0):
            temp = data
        else:
            temp = np.vstack((temp, data))
    return temp


def pca(x, k):
    n_samples, n_features = x.shape
    # 求均值
    mean = np.array([np.mean(x[:, i]) for i in range(n_features)])
    # 去中心化
    normal_data = x - mean
    Var = np.dot(np.transpose(normal_data), normal_data)
    w, v = np.linalg.eig(Var)
    v = np.real(v)
    w = np.real(w)
    #求出前k个特征向量
    eigIndex = np.argsort(w)
    eigVecIndex = eigIndex[:-(k + 1):-1]
    feature = v[:, eigVecIndex]
    new_x = np.dot(normal_data, feature)
    rec = np.dot(new_x, feature.T)+mean
    return rec

def psnr(img1, img2):
   mse = np.mean((img1/255. - img2/255.) ** 2)
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def drawDimen3(X, w, v):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = X.shape[0]
    for i in range(n):
        ax.scatter(X[i, 0], X[i, 1], X[i, 2])
    plt.show()
    X1 = np.dot(X, v.T[:, 0:2])
    for i in range(n):
        plt.plot(X1[i, 0], X1[i, 1], "go")
    plt.show()


if __name__ == '__main__':
    # X = create_data()
    # pca(X, 2)
    # Var = calculate_covariance_matrix(X)
    # # print(Var)
    # w, v = eig(Var)
    # drawDimen3(X, w, v)
    # data = loadImage()
    # new_data = pca(data, 10)
    # newImage = Image.fromarray(new_data*100)
    # newImage.show()
    # print(new_data.shape)
    N = 10
    data = loadImages()
    for i in range(N):
        plt.subplot(2, 5, i+1)
        plt.imshow(data[i].reshape(30, 30))
    plt.show()
    rec = pca(data, 1)
    for i in range(N):
        plt.subplot(2, 5, i+1)
        rec1 = rec[i, :]
        data1 = rec1.reshape(30, 30)
        plt.imshow(data1)
    plt.show()
    for i in range(N):
        print(psnr(data[i].reshape(30, 30), rec[i].reshape(30, 30)))




