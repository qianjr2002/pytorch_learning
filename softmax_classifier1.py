import torch
import torch.nn.functional as F

# 定义张量
y = torch.tensor([1, 0, 0], dtype=torch.float32)
z = torch.tensor([0.2, 0.1, -0.1], dtype=torch.float32)

# 计算预测概率分布
y_pred = F.softmax(z, dim=0)

# 计算负对数似然损失
loss = -torch.sum(y * torch.log(y_pred))

print(loss.item())
