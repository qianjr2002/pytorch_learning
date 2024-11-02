import torch
import numpy as np

# 原始数据
data = np.array([[1, 2, 3, 4, 1, 2, 3, 4], [ 1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.float32)
print(f'data.shape = {data.shape}')
print(data)

# 使用 n_fft=4，hop_length=2，win_length=4
input1 = torch.stft(torch.from_numpy(data), n_fft=4, hop_length=2, win_length=4, return_complex=True)
# window=torch.hann_window(4).pow(0.5), 
print("return_complex=True")
print(f'input1 = {input1}')
print(f'input1.size = {input1.size()}')

input2 = torch.stft(torch.from_numpy(data), n_fft=4, hop_length=2, win_length=4, window=torch.hann_window(4).pow(0.5), return_complex=False)
print("return_complex=False")
print(f'input2 = {input2}')
print(f'input2.size = {input2.size()}')