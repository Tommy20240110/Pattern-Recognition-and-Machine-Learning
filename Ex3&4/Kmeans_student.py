import os
import random
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_euclidean_distance(x1, x2):
    """
    计算两个样本之间的欧氏距离
    参数:
        x1, x2: 两个向量
    返回:
        欧氏距离
    """
    distance = 0
    # 距离的平方项再开根号
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return np.sqrt(distance)


def initialize_centroids(num_clusters, data):
    """
    随机初始化 k 个聚类中心
    参数:
        num_clusters: 聚类个数 k
        data: 数据集 (n_samples, n_features)
    返回:
        初始化的中心点矩阵 (k, n_features)
    """
    n_samples, n_features = data.shape
    centroids = np.zeros((num_clusters, n_features))
    for i in range(num_clusters):
        # 每一次循环随机选择一个类别中心
        centroid = data[np.random.choice(range(n_samples))]
        centroids[i] = centroid
    return centroids


def find_closest_centroid(sample, centroids):
    """
    找到样本最近的聚类中心索引
    参数:
        sample: 单个样本
        centroids: 当前所有中心点
    返回:
        最近中心点的索引
    """
    closest_index = 0
    min_distance = float("inf")
    for i, centroid in enumerate(centroids):
        # 根据欧式距离判断, 选择最小距离的中心点所属类别
        distance = compute_euclidean_distance(sample, centroid)
        if distance < min_distance:
            closest_index = i
            min_distance = distance
    return closest_index


def assign_samples_to_clusters(centroids, num_clusters, data):
    """
    根据当前中心点将样本分配到最近的簇
    参数:
        centroids: 当前中心点
        num_clusters: 聚类个数 k
        data: 数据集
    返回:
        clusters: 列表, 每个元素是该簇中样本的索引列表
    """
    clusters = [[] for _ in range(num_clusters)]
    for sample_index, sample in enumerate(data):
        # 将样本划分到最近的类别区域
        centroid_index = find_closest_centroid(sample, centroids)
        clusters[centroid_index].append(sample_index)
    return clusters


def update_centroids(clusters, num_clusters, data):
    """
    计算新的聚类中心
    参数:
        clusters: 当前聚类结果
        num_clusters: 聚类个数 k
        data: 数据集
    返回:
        新的中心点矩阵
    """
    n_features = np.shape(data)[1]
    new_centroids = np.zeros((num_clusters, n_features))
    # 实验一: 以当前每个类样本的均值为新的中心点
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            # 取簇内所有样本的均值作为新中心点
            new_centroid = np.mean(data[cluster], axis=0)
        else:
            # 如果簇为空，随机选择一个样本作为新中心
            random_index = np.random.choice(range(data.shape[0]))
            new_centroid = data[random_index]
        new_centroids[i] = new_centroid
    return new_centroids


def get_cluster_labels(clusters, data):
    """
    获取每个样本所属的簇标签
    参数:
        clusters: 聚类结果
        data: 数据集
    返回:
        y_pred: 每个样本的标签数组
    """
    y_pred = np.zeros(np.shape(data)[0])
    for cluster_index, cluster in enumerate(clusters):
        for sample_index in cluster:
            y_pred[sample_index] = cluster_index
    return y_pred


def kmeans(data, num_clusters, max_iterations):
    """
    K-means 算法流程
    参数:
        data: 数据集
        num_clusters: 聚类个数 k
        max_iterations: 最大迭代次数
    返回:
        聚类标签
    """
    # 1. 初始化中心点
    centroids = initialize_centroids(num_clusters, data)
    # 遍历迭代求解
    for iteration in range(max_iterations):
        # 2. 根据当前中心点进行聚类
        clusters = assign_samples_to_clusters(centroids, num_clusters, data)
        # 保存当前中心点
        prev_centroids = centroids
        # 3.根据聚类结果计算新的中心点
        centroids = update_centroids(clusters, num_clusters, data)
        # 实验二: 打印每个 interation 的中心, 观察其变化
        print(f"=== 第 {iteration + 1} 次迭代 ===")
        print(f"当前中心点:\n{prev_centroids}")
        print(f"新中心点:\n{centroids}\n")
        # 4. 检查收敛条件直到中心点不再变化
        diff = centroids - prev_centroids
        if not diff.any():
            break
    # 返回最终的聚类标签
    return get_cluster_labels(clusters, data)


if __name__ == "__main__":
    ex3_path = os.path.dirname(__file__)
    # 简单数据集测试
    print("=== 简单数据集测试 ===")
    X = np.array([[0, 2], [0, 0], [1, 0], [5, 0], [5, 2]])
    # 设定聚类类别为 2 个, 最大迭代次数为 10 次
    labels = kmeans(X, 2, 10)
    # 打印每个样本所属的类别标签
    print(f"最终聚类标签: {labels}\n")

    # 3D 数据集测试
    print("=== 3D 数据集测试 ===")
    X, y = datasets.make_blobs(
        n_samples=10000,
        n_features=3,
        centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
        cluster_std=[0.2, 0.1, 0.2, 0.2],
        random_state=9,
    )

    # 用 Kmeans 算法进行聚类
    # 实验三: 选择不同的 K 进行聚类, 观察结果并画图
    NUM_CLUSTERS = 4
    MAX_ITER = 100
    blobs_labels = kmeans(X, NUM_CLUSTERS, MAX_ITER)
    print(f"最终聚类标签: {blobs_labels}\n")
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for cluster_i in range(NUM_CLUSTERS):
        ax.scatter(
            X[blobs_labels == cluster_i][:, 0],
            X[blobs_labels == cluster_i][:, 1],
            X[blobs_labels == cluster_i][:, 2],
        )
    plt.show()
    fig1_path = ex3_path + "\\Ex3_Fig1.jpg"
    plt.savefig(fig1_path)
