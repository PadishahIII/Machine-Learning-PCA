import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random.mtrand import RandomState
import os
import matplotlib.image as mpimg
import cv2
from PIL import Image
import math

size = (40, 40)  #将图像压缩成size大小


def PCA(data, k):
    row, col = data.shape
    mean = np.sum(data, 0) / row  #均值
    data_centered = data - mean  #均一化
    cov = np.dot(data_centered.T, data_centered)  #协方差矩阵
    eigValues, eigVectors = np.linalg.eig(cov)  #求协方差矩阵的特征值和特征向量
    eigValueSorted = np.argsort(eigValues)  #特征值排序
    eigVectorTar = eigVectors[:,
                              eigValueSorted[:-(k + 1):-1]]  #提取最大的k个特征值对应的特征向量
    return data_centered, eigVectorTar, mean


def show_3D(X):
    fig = plt.figure()
    axes = Axes3D(fig)
    axes.view_init(elev=20, azim=80)
    axes.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 0], cmap=plt.cm.gnuplot)
    axes.legend(loc='best')
    plt.show()


def show_2D(X):
    plt.scatter(X[:, 0].tolist(),
                X[:, 1].tolist(),
                c=X[:, 0].tolist(),
                cmap=plt.cm.gnuplot)
    plt.show()


def run_PCA_data(X):
    show_3D(X)
    data_centered, eigVectors, mean = PCA(X, 2)
    data_transformed = np.dot(data_centered, eigVectors)
    show_2D(data_transformed)


'''
对x进行旋转
theta:旋转的弧度
axis:所绕的轴，x,y,z
'''


def rotate(X, theta=0, axis='x'):
    if axis == 'x':
        rotate = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]]
        return np.dot(rotate, X)
    elif axis == 'y':
        rotate = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]]
        return np.dot(rotate, X)
    elif axis == 'z':
        rotate = [[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        return np.dot(rotate, X)
    else:
        print('Invaild axis')
        return X


"""
n:数据点个数
noise:噪声程度
scale_y:厚度
"""


def generate(n=100, noise=0.0, scale_y=100):
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(1, n))
    x = t * np.cos(t)
    y = scale_y * np.random.rand(1, n)
    z = t * np.sin(t)
    X = np.concatenate((x, y, z))
    X += noise * np.random.randn(3, n)
    X = rotate(X, 40 * np.pi / 180, 'z')
    X = X.T
    return X


x1 = generate(1500, 0, 20)
#x2 = generate(1000, 0, 10)
#x3 = generate(1000, 1, 10)
run_PCA_data(x1)
#run_PCA_data(x2)
#run_PCA_data(x3)
'''
read data of face from path
'''


def read_face_data(file_path):
    file_list = os.listdir(file_path)
    data = []
    i = 1
    plt.figure(figsize=size)
    for file in file_list:
        path = os.path.join(file_path, file)
        plt.subplot(3, 4, i)
        with open(path) as f:
            img = cv2.imread(path)  #read img data
            img = cv2.resize(img, size)  #压缩图像
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #灰度图
            plt.imshow(img_gray)
            h, w = img_gray.shape
            img_col = img_gray.reshape(h * w)  #拉平
            data.append(img_col)
        i += 1
    plt.show()
    return np.array(data)


'''
计算峰值信噪比
'''


def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.)**2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


data = read_face_data('data')
n_sample, n_features = data.shape
data_centered, eigVectors, mean_data = PCA(data, 1)
print(eigVectors)
eigVectors = np.real(eigVectors)  #特征向量矩阵可能出现复向量，保留实部
data_pca = np.dot(data_centered, eigVectors)
data_reconstruct = np.dot(data_pca, eigVectors.T) + mean_data
plt.figure(figsize=size)
for i in range(n_sample):
    plt.subplot(3, 4, i + 1)
    plt.imshow(data_reconstruct[i].reshape(size))
plt.show()

print("信噪比：")
for i in range(n_sample):
    a = psnr(data[i], data_reconstruct[i])
    print("img ", i, " 信噪比:", a)
