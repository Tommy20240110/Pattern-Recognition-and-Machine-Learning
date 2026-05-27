import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class PCA():
    # 计算协方差矩阵
    def calculate_covariance_matrix(self, X):
        m = X.shape[0]
        # 数据标准化
        X = X - np.mean(X, axis=0) # 当axis=0时，我们沿着每一列或行标签向下执行，而axis=1时，我们沿着每一行或者列标签向右执行。
        return ??        # 扩展实验：协方差矩阵的计算
    # matmul()函数：矩阵乘法

    def pca(self, X, n_components):
        # 计算协方差矩阵
        covariance_matrix = self.calculate_covariance_matrix(X)
        # 计算协方差矩阵的特征值和对应特征向量
        eigenvalues, eigenvectors =    # 扩展实验：特征值和特征向量

        # 对特征值排序
        idx = eigenvalues.argsort()[::-1]
        # 取最大的前n_component组
        eigenvectors = eigenvectors[:, idx]
        eigenvectors = eigenvectors[:, :n_components]
        # Y=PX转换
        return np.matmul(X, eigenvectors)


from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

# 导入sklearn数据集
data = datasets.load_digits()
X = data.data
y = data.target
X1=X[:, 0]
X2=X[:, 1]

cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

class_distr = []
# 绘制不同类别分别
for i, l in enumerate(np.unique(y)):
    _X1 = X1[y == l]
    _X2 = X2[y == l]
    _y = y[y == l]
    class_distr.append(plt.scatter(_X1, _X2, color=colors[i]))

# 图例
plt.legend(class_distr, y, loc=1)

# 坐标轴
plt.suptitle("PCA Dimensionality Reduction")
plt.title("Digit Dataset")
plt.xlabel('Original Component 1')
plt.ylabel('Original Component 2')
plt.show()


# 将数据降维到2个主成分
X_trans = PCA().pca(X, 2)
x1 = X_trans[:, 0]
x2 = X_trans[:, 1]


cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

class_distr = []
# 绘制不同类别分别
for i, l in enumerate(np.unique(y)):
    _x1 = x1[y == l]
    _x2 = x2[y == l]
    _y = y[y == l]
    class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

# 图例
plt.legend(class_distr, y, loc=1)

# 坐标轴
plt.suptitle("PCA Dimensionality Reduction")
plt.title("Digit Dataset")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
