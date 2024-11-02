import torch
criterion = torch.nn.CrossEntropyLoss()
Y = torch.LongTensor([2, 0, 1])

Y_pred1 = torch.tensor([[0.1, 0.2, 0.9],
                        [1.1, 0.1, 0.2],
                        [0.2, 2.1, 0.1]
                        ],dtype=torch.float32)

Y_pred2 = torch.tensor([[0.8, 0.2, 0.3],
                        [0.2, 0.3, 0.5],
                        [0.2, 0.2, 0.5]
                        ],dtype=torch.float32)

loss1 =criterion(Y_pred1, Y)
print('loss1 =',loss1)

loss2 =criterion(Y_pred2, Y)
print('loss2 =',loss2)