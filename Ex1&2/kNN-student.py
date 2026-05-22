# 1.  python package的导入：

import numpy as np # 导入numpy package
import pandas as pd # 导入pandas package
from sklearn.datasets import load_iris # 导入经典数据库iris
import matplotlib.pyplot as plt  # 导入绘图工具pyplot package
from collections import Counter  # 列表、字典、字符串等中计算元素重复的次数

# 2. 数据的导入：以 iris 数据集为例;
data = load_iris()
X=data.data  # 导入数据X shape:(150, 4)  # 代表有150条数据，每条数据有4个特征
y=data.target  # 导入标签Y shape:(150, ) # 代表150条数据的标签

#实验1：观察数据集--使用print功能

print(data)

# 3. 数据的可视化
plt.figure(1, figsize=(8, 6))  # 1是图片编号 8和6是宽和高
plt.clf()  # 清空当前的图

# 可视化训练数据
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')  # 可视化特征1和特征2
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

# 实验2：调整画图中的参数-- colormap，edgecolor， marker等

# 可视化标签数据
plt.figure(2, figsize=(8, 6))
plt.scatter(X[:, 0], y, cmap=plt.cm.Set2,
            edgecolor='k')  # 可视化特征1和类别y
plt.xlabel('Sepal length')
plt.ylabel('class')
plt.show()

#4. 训练集与测试集的划分
offset = int(X.shape[0]*0.7)  # shape[0]：矩阵的行数  shape[1]：矩阵的列数
X_train, y_train = X[:offset], y[:offset] # 取前70% 为训练集
X_test, y_test = X[offset:], y[offset:]  # 取后30% 为测试集
y_train = y_train.reshape(-1,1)  # reshape(1,-1)：矩阵转换为一行    reshape(-1,1)：矩阵转换为一列
y_test = y_test.reshape(-1,1)

print('X_train=',X_train.shape)
print('X_test=',X_test.shape)
print('y_train=',y_train.shape)
print('y_test=',y_test.shape)

#5. 定义距离度量函数：欧式距离
def compute_distance (X_test, X_train):
    te = np.square(X_test).sum(axis=1) # 计算y的平方 shape:(45, )
    tr = np.square(X_train).sum(axis=1) # 计算x的平方 shape:(105, )
    M = np.dot(X_test,X_train.T) # 计算x和y相乘 shape:(45, 105)
    dists = np.sqrt(-2 * M + tr + np.mat(te).T) # x的平方 + y的平方 - 2xy Python广播机制 平方根
    return dists

# 扩展实验1：定义曼哈顿距离

# 我们尝试计算一下测试集与训练集实例的距离：
dist_test = compute_distance(X_test, X_train) # shape:(45， 105)
print ('距离',dist_test.shape)

#  顺便对距离矩阵进行可视化展示一下：

plt.figure(2, figsize=(8, 6))
plt.xlabel('Training sample')
plt.ylabel('Testing sample')
plt.imshow(dist_test,interpolation='none',cmap=plt.cm.gray)


plt.show()

#6. 使用多数表决的分类决策规则定义预测函数，这里假设 k 值取 1：
def predict_labels(y_train, dists, k=1):
    num_test=dists.shape[0] #获得测试样本的数量
    y_pred=np.zeros(num_test)  # 初始化预测值 y_pred
    for i in range(num_test):
        labels = y_train[np.argsort(dists[i,:])].flatten() # 将每个测试数据到训练数据的label按距离大小排序
        # argsort(): 将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
        # flatten(): 返回一个折叠成一维的数组
        closest_y = labels[0:k]   # 选出前k个近邻点的label
        c=Counter(closest_y) # 计算k个近邻点的label的数目
        y_pred[i]=c.most_common(1)[0][0]  # 选出多数表决法的类别
    return y_pred

## 跑一下预测测试集的类别准确率：
y_test_pred = predict_labels(y_train, dist_test, k=1)
y_test_pred = y_test_pred.reshape(-1,1)  # 换成一列的形式
num_correct = np.sum(y_test_pred==y_test)  # 计算预测正确的数目
accuracy = num_correct/X_test.shape[0]  # 计算正确率
print('accuracy is ',accuracy)

##7. 拓展：尝试使用5折交叉验证来选择最优的k值：
num_folds = 5   # 5折交叉验证
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)
k_to_accuracies = {}
for k in k_choices:
    for fold in range(num_folds):
        # 对传入的训练集单独划出一个验证集作为测试集
        validation_X_test = X_train_folds[fold]
        validation_y_test = y_train_folds[fold]
        temp_X_train = np.concatenate(X_train_folds[:fold] + X_train_folds[fold + 1:])
        temp_y_train = np.concatenate(y_train_folds[:fold] + y_train_folds[fold + 1:])
        # 计算距离
        temp_dists = compute_distance(validation_X_test,temp_X_train)
        temp_y_test_pred = predict_labels(temp_y_train,temp_dists, k=k)
        temp_y_test_pred = temp_y_test_pred.reshape((-1, 1))
        # 查看分类准确率
        num_correct = np.sum(temp_y_test_pred == validation_y_test)
        num_test = validation_X_test.shape[0]
        accuracy = float(num_correct) / num_test
        k_to_accuracies[k] = k_to_accuracies.get(k, []) + [accuracy]

## 打印不同 k 值不同折数下的分类准确率
for k in k_choices:
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))

for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()


##查看最佳k值大小：
best_k = k_choices[np.argmax(accuracies_mean)]
print('最佳k值为',best_k)