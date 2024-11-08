import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0]])

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegressionModel()

# 损失函数和优化器
criterion = nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0 or epoch == 999:  # 每100个epoch打印一次损失值
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# 切换到评估模式
model.eval()

# 绘制模型的预测曲线
x = np.linspace(0, 10, 200)
x_t = torch.tensor(x, dtype=torch.float32).view((200, 1))

# 在评估时不需要梯度
with torch.no_grad():  
    y_t = model(x_t)
    y = y_t.data.numpy()

plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of pass')
plt.grid()
plt.show()
