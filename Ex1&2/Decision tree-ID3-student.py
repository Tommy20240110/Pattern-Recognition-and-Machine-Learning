import os
import numpy as np
import pandas as pd
from math import log




# 引入数据集
# 绝对路径
# df = pd.read_csv('C:\Python Project\Machine Learning Code Demo\Machine Learning Demo\student-performance.csv')
# 相对路径
df = pd.read_csv('student-performance.csv')
print("data:\n", df,'\n')

def entropy(ele):
    #函数: 计算熵值
    #输入: 训练数据集
    #输出: 熵值
    #计算公式 = - sum(p * log(p)), p是概率.

    # 计算随机变量的分布概率
    probs = [ele.count(i)/len(ele) for i in set(ele)]   # set的数据类型可以去除去除其中重复的元素
    entropy = -sum([prob*log(prob,2) for prob in probs])
    return entropy

def split_dataframe(data,col):
    #函数: 将数据集划分为相应的类别
    #输入: 数据集，特征名
    #输出: 划分好的数据子集

    # 每一个属性的取值
    unique_value =data[col].unique()  #对于一维数组或者列表，unique函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表
    # 数据集的空集
    result_dict = {elem: pd.DataFrame for elem in unique_value}   # 将类别放入dictionary
    # 将数据集按类别划分
    for key in result_dict.keys():
        result_dict[key] = data[:][data[col]==key]  # 将数据分入到不同的类别  [:]代表所有的行， [col]==key代表某一个特征的那一列
        # data[][]二维数组取值
    return result_dict



def choose_best_col(df,label):
    #函数：根据信息增益划分选择出相应的特征
    #输入: 数据集, 标签
    #输出: 最大信息增益, 最佳特征,
         #   根据最佳特征划分数据集.

    # 计算类别的熵值
    entropy_D = entropy(df[label].tolist())  # tolist()用于将数组或矩阵转换成列表
    cols = [col for col in df.columns if col not in [label]]  # 去除不是class/label的特征集 使用[label]是为了能处理假设一条数据有多个标签的情况 eg.['Pass', 'class']

    # 初始化最大信息增益，最佳特征和最佳划分子集
    max_value, best_col = -999, None
    max_splited = None
    # 划分数据，对每一个数据特征进行循环处理
    for col in cols:
        splited_set = split_dataframe(df,col)   # 根据类别划分数据集
        entropy_DA=0   # 初始化
        for subset_col, subset in splited_set.items():   # items()是将dict转换为list
            # 计算分类数据集的标签的熵
            entropy_Di = entropy(subset[label].tolist())
            # 计算当前特征的条件熵

            entropy_DA +=

        # 计算每个特征的信息增益

        info_gain =
        if info_gain>max_value:   # 计算最大信息增益
            max_value, best_col = info_gain, col
            max_splited = splited_set
    return max_value, best_col, max_splited

### 扩展实验2：计算每个特征的信息增益比



# 因为返回的内容包括多种数据格式，打印控制台不分行可能不太好看
# print(choose_best_col(df,'Pass'))

if __name__ == '__main__':
    max_value, best_col, max_splited = choose_best_col(df, 'Pass')
    print("max value:", max_value)
    print("best col:", best_col)
    print("max splited:")
    for i in max_splited.items():
        print(i, '\n')




