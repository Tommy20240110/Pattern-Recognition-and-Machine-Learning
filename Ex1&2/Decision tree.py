import os
import numpy as np
import pandas as pd
from math import log


def compute_entropy(label_col):
    """
    计算熵值
    参数:
        label_col: 标签列数据
    返回:
        熵值
    公式: H(D) = - sum(p * log2(p))
    """
    # 计算每个类别出现的概率
    label_list = label_col.tolist() if hasattr(label_col, "tolist") else list(label_col)
    unique_labels = set(label_list)
    probs = [label_list.count(label) / len(label_list) for label in unique_labels]
    # 计算熵
    entropy = -sum([prob * log(prob, 2) for prob in probs])
    return entropy


def split_dataframe(data, feature):
    """
    根据特征将数据集划分为相应的子集
    参数:
        data: 数据集 (DataFrame)
        feature: 特征名称
    返回:
        result_dict: 字典, key为特征取值, value为对应的子数据集
    """
    # 获取特征的所有唯一取值
    unique_values = data[feature].unique()
    # 创建空字典存放划分后的子集
    result_dict = {}
    # 按特征取值划分数据集
    for value in unique_values:
        result_dict[value] = data[data[feature] == value].copy()
    return result_dict


def id3_algo(data, label):
    """
    根据信息增益选择最佳划分特征 (ID3算法)
    参数:
        data: 数据集 (DataFrame)
        label: 标签列名称
    返回:
        max_info_gain: 最大信息增益
        best_feature: 最佳特征名称
        best_splited: 根据最佳特征划分后的子集字典
    """
    # 计算标签的熵 H(D)
    entropy_d = compute_entropy(data[label])
    # 获取所有特征列 (排除标签列)
    features = [col for col in data.columns if col not in [label]]
    # 初始化
    max_info_gain = -float("inf")
    best_feature = None
    best_splited = None
    # 遍历每个特征, 计算信息增益
    for feature in features:
        # 根据当前特征划分数据集
        splited_set = split_dataframe(data, feature)
        # 计算条件熵 H(D|A)
        entropy_da = 0
        for _, subset in splited_set.items():
            if len(subset) > 0:
                # 计算子集的熵
                entropy_di = compute_entropy(subset[label])
                # 加权求和
                entropy_da += len(subset) / len(data) * entropy_di
        # 计算信息增益 g(D,A) = H(D) - H(D|A)
        info_gain = entropy_d - entropy_da
        # 更新最佳特征
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature
            best_splited = splited_set
    return max_info_gain, best_feature, best_splited


def c4_5_algo(data, label):
    """
    根据信息增益比选择最佳划分特征 (C4.5算法)
    参数:
        data: 数据集 (DataFrame)
        label: 标签列名称
    返回:
        max_ratio: 最大信息增益比
        best_feature: 最佳特征名称
        best_splited: 根据最佳特征划分后的子集字典
    """
    # 计算标签的熵 H(D)
    entropy_d = compute_entropy(data[label])
    # 获取所有特征列 (排除标签列)
    features = [col for col in data.columns if col != label]
    # 初始化
    max_ratio = -float("inf")
    best_feature = None
    best_splited = None
    # 遍历每个特征, 计算信息增益比
    for feature in features:
        # 根据当前特征划分数据集
        splited_set = split_dataframe(data, feature)
        # 计算条件熵 H(D|A)
        entropy_da = 0
        # 计算特征 A 的熵 H(A)
        entropy_a = 0
        for _, subset in splited_set.items():
            if len(subset) > 0:
                # 计算子集的熵
                entropy_di = compute_entropy(subset[label])
                # 条件熵加权求和
                entropy_da += len(subset) / len(data) * entropy_di
                # 计算特征 A 的熵 H(A)
                prob = len(subset) / len(data)
                entropy_a += -prob * log(prob, 2) if prob > 0 else 0
        # 计算信息增益
        info_gain = entropy_d - entropy_da
        # 计算信息增益比 g_R(D,A) = g(D,A) / H(A)
        if entropy_a > 0:
            info_gain_ratio = info_gain / entropy_a
        else:
            info_gain_ratio = 0
        # 更新最佳特征
        if info_gain_ratio > max_ratio:
            max_ratio = info_gain_ratio
            best_feature = feature
            best_splited = splited_set
    return max_ratio, best_feature, best_splited


if __name__ == "__main__":
    ex2_path = os.path.dirname(__file__)

    # 1. 数据集
    print("=== 1. 数据集 ===")
    data_path = ex2_path + "\\student-performance.csv"
    data = pd.read_csv(data_path)
    print()

    # 2. 使用信息增益选择最佳特征
    print("=== 2. 使用信息增益选择最佳特征 ===")
    max_gain, best_feature, best_splited = id3_algo(data, "Pass")
    print(f"最大信息增益: {max_gain:.4f}")
    print(f"最佳划分特征: {best_feature}")
    for i in best_splited.items():
        print(i, "\n")

    # 3. 使用信息增益比选择最佳特征
    print("=== 3. 使用信息增益比选择最佳特征 ===")
    max_ratio, best_feature_ratio, best_splited = c4_5_algo(data, "Pass")
    print(f"最大信息增益比: {max_ratio:.4f}")
    print(f"最佳划分特征: {best_feature_ratio}")
    for i in best_splited.items():
        print(i, "\n")
