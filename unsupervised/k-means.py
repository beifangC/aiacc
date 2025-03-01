from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成500个样本，2个特征，4个簇，标准差为0.6
X, y = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=0.6, random_state=42)

# 可视化原始数据
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.title("原始数据分布")
plt.show()


from sklearn.cluster import KMeans

# 初始化模型并设置簇数为4
kmeans = KMeans(n_clusters=4, random_state=42)

# 训练模型并预测簇标签
labels = kmeans.fit_predict(X)


# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker='X', c='red')
plt.title("K-Means聚类结果")
plt.show()

from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
print(f"轮廓系数: {score:.2f}")  # 值越接近1，聚类效果越好