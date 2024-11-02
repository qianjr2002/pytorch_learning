import numpy as np

y = np.array([1, 0, 0])
z = np.array([0.2, 0.1, -0.1])

# 计算预测概率分布
y_pred = np.exp(z) / np.exp(z).sum()

# 计算负对数似然损失
loss = -np.sum(y * np.log(y_pred))

print(loss)