from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据（乳腺癌二分类数据集）
data = load_breast_cancer()
X, y = data.data, data.target


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建随机森林分类器
rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# 预测与评估
y_pred = rf.predict(X_test)
print("准确率:", accuracy_score(y_test, y_pred))

# 特征重要性（显示前5个重要特征）
feature_importance = dict(zip(data.feature_names, rf.feature_importances_))
sorted_features = sorted(feature_importance.items(), key=lambda x: -x[1])
print("前5个重要特征:", sorted_features[:5])