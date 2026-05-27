import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import matplotlib.cm as cmx
import matplotlib.colors as colors


class PCA:
    """
    主成分分析 (Principal Component Analysis)
    用于数据降维
    """
    
    def calculate_covariance_matrix(self, X):
        """
        计算协方差矩阵
        参数:
            X: 数据集 (n_samples, n_features)
        返回:
            协方差矩阵 (n_features, n_features)
        """
        # 数据标准化（零均值化）
        X_centered = X - np.mean(X, axis=0)
        n_samples = X_centered.shape[0]
        # 协方差矩阵 = (1/(n-1)) * X^T * X
        covariance_matrix = np.matmul(X_centered.T, X_centered) / (n_samples - 1)
        return covariance_matrix
    
    def calculate_covariance_matrix_by_np(self, X):
        """
        使用 numpy 内置函数计算协方差矩阵（备用方法）
        参数:
            X: 数据集 (n_samples, n_features)
        返回:
            协方差矩阵 (n_features, n_features)
        """
        return np.cov(X, rowvar=False)  # rowvar=False 表示每列是一个特征
    
    def pca(self, X, n_components):
        """
        执行 PCA 降维
        参数:
            X: 数据集 (n_samples, n_features)
            n_components: 降维后的维度（主成分个数）
        返回:
            降维后的数据 (n_samples, n_components)
        """
        # 1. 数据零均值化
        X_centered = X - np.mean(X, axis=0)
        
        # 2. 计算协方差矩阵
        covariance_matrix = self.calculate_covariance_matrix(X)
        
        # 3. 计算协方差矩阵的特征值和对应特征向量
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        
        # 4. 对特征值降序排序，取前 n_components 个特征向量
        idx = np.argsort(eigenvalues)[::-1]  # 从大到小排序的索引
        eigenvectors = eigenvectors[:, idx]   # 特征向量按特征值大小排序
        eigenvectors = eigenvectors[:, :n_components]  # 取前 n_components 个
        
        # 5. 数据投影变换: Y = X * W
        X_transformed = np.matmul(X_centered, eigenvectors)
        
        return X_transformed
    
    def get_explained_variance_ratio(self, X):
        """
        计算各主成分的方差贡献率（用于选择 n_components）
        参数:
            X: 数据集 (n_samples, n_features)
        返回:
            方差贡献率数组
        """
        # 计算协方差矩阵
        covariance_matrix = self.calculate_covariance_matrix(X)
        # 计算特征值
        eigenvalues, _ = np.linalg.eig(covariance_matrix)
        # 对特征值降序排序
        eigenvalues = np.sort(eigenvalues)[::-1]
        # 计算方差贡献率
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        return explained_variance_ratio
    
    def plot_cumulative_variance(self, X, fig_path=None):
        """
        绘制累积方差贡献率图，用于确定最佳降维维度
        参数:
            X: 数据集
            fig_path: 图片保存路径（可选）
        """
        explained_variance_ratio = self.get_explained_variance_ratio(X)
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-', linewidth=2)
        plt.axhline(y=0.8, color='r', linestyle='--', label='80% 累积贡献率')
        plt.axhline(y=0.9, color='g', linestyle='--', label='90% 累积贡献率')
        plt.xlabel('主成分个数')
        plt.ylabel('累积方差贡献率')
        plt.title('PCA 累积方差贡献率')
        plt.legend()
        plt.grid(True)
        
        if fig_path:
            plt.savefig(fig_path)
        plt.show()
        return cumulative_variance


def plot_scatter_with_labels(data, class_labels, title, xlabel, ylabel, fig_path=None):
    """
    绘制带标签的散点图
    参数:
        data: 数据集 (n_samples, 2)
        class_labels: 类别标签数组
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        fig_path: 图片保存路径（可选）
    """
    plt.figure(figsize=(8, 6))
    
    # 获取颜色映射
    cmap = plt.get_cmap('viridis')
    unique_classes = np.unique(class_labels)
    colors = [cmap(i) for i in np.linspace(0, 1, len(unique_classes))]
    
    # 绘制不同类别的散点图
    class_distributions = []
    for idx, class_value in enumerate(unique_classes):
        x_data = data[class_labels == class_value][:, 0]
        y_data = data[class_labels == class_value][:, 1]
        scatter = plt.scatter(x_data, y_data, color=colors[idx], alpha=0.7, s=20, label=f'类别 {class_value}')
        class_distributions.append(scatter)
    
    # 添加图例
    plt.legend(loc=1, title="类别")
    
    # 坐标轴和标题
    plt.suptitle("PCA 主成分分析降维")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    
    if fig_path:
        plt.savefig(fig_path)
    plt.show()


if __name__ == "__main__":
    # ==================== 加载数据集 ====================
    print("=" * 50)
    print("PCA 主成分分析实验")
    print("=" * 50)
    
    # 加载手写数字数据集
    digit_data = load_digits()
    X = digit_data.data      # 特征数据 (1797, 64)
    y = digit_data.target    # 标签 (0-9)
    
    print(f"原始数据维度: {X.shape}")
    print(f"类别数量: {len(np.unique(y))}")
    
    # ==================== 原始数据可视化（前两个特征） ====================
    print("\n--- 原始数据可视化（前两个特征维度）---")
    plot_scatter_with_labels(
        data=np.column_stack((X[:, 0], X[:, 1])),
        class_labels=y,
        title="原始数据（前两个特征）",
        xlabel="特征 1",
        ylabel="特征 2"
    )
    
    # ==================== PCA 降维 ====================
    print("\n--- PCA 降维处理 ---")
    pca_model = PCA()
    
    # 计算累积方差贡献率，用于选择合适的主成分个数
    print("计算各主成分方差贡献率...")
    explained_variance_ratio = pca_model.get_explained_variance_ratio(X)
    print(f"各主成分方差贡献率: {explained_variance_ratio[:10]}")  # 打印前10个
    print(f"前2个主成分累积贡献率: {np.sum(explained_variance_ratio[:2]):.4f}")
    print(f"前10个主成分累积贡献率: {np.sum(explained_variance_ratio[:10]):.4f}")
    
    # 绘制累积方差贡献率图
    print("绘制累积方差贡献率图...")
    cumulative_variance = pca_model.plot_cumulative_variance(X)
    
    # 将数据降维到 2 个主成分
    print("将数据降维到 2 个主成分...")
    X_transformed = pca_model.pca(X, n_components=2)
    print(f"降维后数据维度: {X_transformed.shape}")
    
    # ==================== 降维结果可视化 ====================
    print("\n--- 降维结果可视化 ---")
    plot_scatter_with_labels(
        data=X_transformed,
        class_labels=y,
        title="PCA 降维结果（2个主成分）",
        xlabel="第一主成分",
        ylabel="第二主成分"
    )
    
    # ==================== 实验：对比不同降维维度的效果 ====================
    print("\n--- 实验：对比不同降维维度 ---")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    n_components_list = [2, 5, 10, 15, 20, 30]
    cmap = plt.get_cmap('viridis')
    unique_classes = np.unique(y)
    colors = [cmap(i) for i in np.linspace(0, 1, len(unique_classes))]
    
    for idx, n_comp in enumerate(n_components_list):
        X_pca = pca_model.pca(X, n_components=n_comp)
        
        # 如果是2维就画散点图，否则画前两维的散点图
        ax = axes[idx]
        for class_idx, class_value in enumerate(unique_classes):
            ax.scatter(
                X_pca[y == class_value][:, 0],
                X_pca[y == class_value][:, 1] if X_pca.shape[1] > 1 else np.zeros_like(X_pca[y == class_value][:, 0]),
                color=colors[class_idx],
                alpha=0.6,
                s=10,
                label=f'类别 {class_value}' if idx == 0 else ""
            )
        ax.set_title(f'主成分个数 = {n_comp}')
        ax.set_xlabel('PC1')
        if X_pca.shape[1] > 1:
            ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)
    
    # 只在第一个子图显示图例
    axes[0].legend(loc='best', fontsize=8)
    
    plt.suptitle("实验：不同主成分个数的降维效果对比", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\n=== PCA 实验完成 ===")

    ex4_path = os.path.dirname(__file__)
    fig1_path = ex4_path + "\\Ex4_Fig1.jpg"
    plt.savefig(fig1_path)