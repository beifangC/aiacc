import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 线性回归的例子，使用了sklearn的数据


# 加载数据
diabetes = load_diabetes()
X = diabetes.data  # 特征矩阵 (442个样本 x 10个特征)
y = diabetes.target  # 目标值

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 线性模型，均方误差作为损失函数，梯度下降作为反向传播
model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# 计算均方误差和R²分数
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# 输出权重和截距
print("系数:", model.coef_)
print("截距:", model.intercept_)


plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # 绘制理想对角线
plt.xlabel("真实值")
plt.ylabel("预测值")
plt.title("真实值 vs 预测值")
plt.show()