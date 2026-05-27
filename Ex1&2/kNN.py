import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris


def compute_euclidean_distance(X_test, X_train):
    """
    计算测试集与训练集之间的欧氏距离矩阵
    使用广播方法加速计算: d^2 = |x|^2 + |y|^2 - 2xy
    参数:
        X_test: 测试集 (n_test_samples, n_features)
        X_train: 训练集 (n_train_samples, n_features)
    返回:
        距离矩阵 (n_test_samples, n_train_samples)
    """
    test_squared = np.square(X_test).sum(axis=1)  # 计算 y 的平方 shape:(45,)
    train_squared = np.square(X_train).sum(axis=1)  # 计算 x 的平方 shape:(105,)
    dot_product = np.dot(X_test, X_train.T)  # 计算 x 和 y 相乘 shape:(45, 105)
    distances = np.sqrt(-2 * dot_product + train_squared + np.asmatrix(test_squared).T)
    return distances


def compute_manhattan_distance(X_test, X_train):
    """
    拓展实验 1: 计算测试集与训练集之间的曼哈顿距离
    参数:
        X_test: 测试集 (n_test_samples, n_features)
        X_train: 训练集 (n_train_samples, n_features)
    返回:
        距离矩阵 (n_test_samples, n_train_samples)
    """
    # 扩展维度以便广播计算
    test_squared = X_test[:, np.newaxis, :]  # shape: (n_test, 1, n_features)
    train_squared = X_train[np.newaxis, :, :]  # shape: (1, n_train, n_features)
    # 计算曼哈顿距离
    distances = np.sum(np.abs(test_squared - train_squared), axis=2)
    return distances


def knn_algo(y_train, distances, k=1):
    """
    使用 k-NN 算法预测标签
    参数:
        y_train: 训练集标签 (n_train_samples, 1)
        distances: 距离矩阵 (n_test_samples, n_train_samples)
        k: 近邻个数
    返回:
        预测标签数组 (n_test_samples,)
    """
    num_test_samples = distances.shape[0]
    y_pred = np.zeros(num_test_samples)
    for i in range(num_test_samples):
        # 获取前 k 个最近邻的标签
        k_nearest_labels = y_train[
            np.argsort(distances[i, :])
        ].flatten()  # 将每个测试数据到训练数据的 label 按距离大小排序
        # argsort(): 将 x 中的元素从小到大排列, 提取其对应的 index, 然后输出到 y
        # flatten(): 返回一个折叠成一维的数组
        # 计算 k 个近邻点的 label 的数目
        label_counter = Counter(k_nearest_labels[0:k])
        y_pred[i] = label_counter.most_common(1)[0][0]
    return y_pred


if __name__ == "__main__":
    ex1_path = os.path.dirname(__file__)

    # 1. 鸢尾花数据集
    print("=== 1. 鸢尾花数据集 ===")
    iris_data = load_iris()
    X = iris_data.data  # 导入数据 X shape: (150, 4)
    y = iris_data.target  # 导入标签 Y shape: (150,)

    # 实验 1: 观察数据集 -- 使用 print 功能
    print(f"特征数据 X:\n{X}")
    print(f"标签数据 y:\n{y}")

    # 2. 数据可视化
    print("\n=== 2. 数据可视化 ===")
    # 可视化特征 1 和特征 2
    plt.figure(1, figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    fig1_path = ex1_path + "\\figures\\Ex1_Fig1.jpg"
    plt.savefig(fig1_path)
    plt.show()
    # 实验 2: 调整画图中的参数 -- colormap, edgecolor, marker 等
    # 可视化特征 1 和类别 y
    plt.figure(2, figsize=(8, 6))
    plt.scatter(X[:, 0], y, cmap=plt.cm.Set2, edgecolor="k")
    plt.xlabel("Sepal length")
    plt.ylabel("class")
    fig2_path = ex1_path + "\\figures\\Ex1_Fig2.jpg"
    plt.savefig(fig2_path)
    plt.show()

    # 3. 划分训练集和测试集
    print("\n=== 3. 划分训练集和测试集 ===")
    offset = int(X.shape[0] * 0.7)  # shape[0]: 矩阵的行数; shape[1]: 矩阵的列数
    X_train, y_train = X[:offset], y[:offset]  # 取前 70% 为训练集
    X_test, y_test = X[offset:], y[offset:]  # 取后 30% 为测试集
    y_train = y_train.reshape(
        -1, 1
    )  # reshape(1, -1): 矩阵转换为一行; reshape(-1, 1): 矩阵转换为一列
    y_test = y_test.reshape(-1, 1)
    print("X_train", X_train.shape)
    print("X_test", X_test.shape)
    print("y_train", y_train.shape)
    print("y_test", y_test.shape)

    # 4. 计算测试集与训练集实例的距离矩阵
    print("\n=== 4. 计算测试集与训练集实例的距离矩阵 ===")
    distances = compute_euclidean_distance(X_test, X_train)
    print("距离", distances.shape)
    # 可视化距离矩阵
    plt.figure(3, figsize=(8, 6))
    plt.xlabel("Training sample")
    plt.ylabel("Testing sample")
    plt.imshow(distances, interpolation="none", cmap=plt.cm.gray)
    fig3_path = ex1_path + "\\figures\\Ex1_Fig3.jpg"
    plt.savefig(fig3_path)
    plt.show()

    #  5. 预测测试集的类别准确率
    print("\n=== 5. 预测测试集的类别准确率 ===")
    y_test_pred = knn_algo(y_train, distances, k=1)
    y_test_pred = y_test_pred.reshape(-1, 1)  # 换成一列的形式
    num_correct = np.sum(y_test_pred == y_test)  # 计算预测正确的数目
    accuracy = num_correct / X_test.shape[0]  # 计算正确率
    print("The accuracy is", accuracy)

    # 6. 拓展: 5 折交叉验证选择最优 k 值
    print("\n=== 6. 交叉验证选择最优 k 值 ===")
    NUM_FOLDS = 5
    K_CHOICES = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    # 将数据分成 NUM_FOLDS 折
    X_train_folds = np.array_split(X_train, NUM_FOLDS)
    y_train_folds = np.array_split(y_train, NUM_FOLDS)
    k_to_accuracies = {}
    # 对每个 k 值进行交叉验证
    for k in K_CHOICES:
        print(f"正在测试 k = {k}...")
        for fold in range(NUM_FOLDS):
            # 划分验证集
            X_val = X_train_folds[fold]
            y_val = y_train_folds[fold]
            # 合并其余折作为训练集
            X_temp_train = np.concatenate(
                X_train_folds[:fold] + X_train_folds[fold + 1 :]
            )
            y_temp_train = np.concatenate(
                y_train_folds[:fold] + y_train_folds[fold + 1 :]
            )
            # 计算距离并预测
            temp_dists = compute_euclidean_distance(X_val, X_temp_train)
            y_pred = knn_algo(y_temp_train, temp_dists, k=k)
            y_pred = y_pred.reshape(-1, 1)
            # 计算准确率
            num_correct = np.sum(y_pred == y_val)
            accuracy = float(num_correct) / X_val.shape[0]
            k_to_accuracies.setdefault(k, []).append(accuracy)
    # 打印交叉验证结果
    print("交叉验证结果:")
    print("-" * 50)
    for k in K_CHOICES:
        for i, accuracy in enumerate(k_to_accuracies[k]):
            print(f"k = {k:3d}, fold = {i+1}, accuracy = {accuracy:.4f}")
        print(
            f"k = {k:3d}, 平均准确率 = {np.mean(k_to_accuracies[k]):.4f} ± {np.std(k_to_accuracies[k]):.4f}"
        )
        print("-" * 50)
    # 绘制交叉验证结果图
    plt.figure(4, figsize=(8, 6))
    # 绘制所有准确率点
    for k in K_CHOICES:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies, alpha=0.6, color="blue")
    # 计算均值和标准差
    accuracies_mean = np.array([np.mean(k_to_accuracies[k]) for k in K_CHOICES])
    accuracies_std = np.array([np.std(k_to_accuracies[k]) for k in K_CHOICES])
    # 绘制误差线
    plt.errorbar(
        K_CHOICES,
        accuracies_mean,
        yerr=accuracies_std,
        fmt="ro-",
        capsize=5,
        linewidth=2,
        markersize=8,
        label="Mean +/- Std",
    )
    plt.title("Cross-validation on k", fontsize=14)
    plt.xlabel("k", fontsize=12)
    plt.ylabel("Cross-validation accuracy", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig4_path = ex1_path + "\\figures\\Ex1_Fig4.jpg"
    plt.savefig(fig4_path)
    plt.show()

    # 找出最佳 k 值
    best_k_index = np.argmax(accuracies_mean)
    best_k = K_CHOICES[best_k_index]
    print(f"\n最佳 k 值为: {best_k} (平均准确率: {accuracies_mean[best_k_index]:.4f})")
