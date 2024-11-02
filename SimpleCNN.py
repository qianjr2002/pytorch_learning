import torch
from ptflops import get_model_complexity_info


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(
            16, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        print('0', x.shape)
        # 0 torch.Size([1, 3, 224, 224])
        x = self.conv1(x)
        print('1', x.shape)
        # 1 torch.Size([1, 16, 224, 224])
        x = self.relu(x)
        print('2', x.shape)
        # 2 torch.Size([1, 16, 224, 224])
        x = self.pool(x)
        print('3', x.shape)
        # 3 torch.Size([1, 16, 112, 112])
        x = self.conv2(x)
        print('4', x.shape)
        # 4 torch.Size([1, 32, 112, 112])
        return x


model = SimpleCNN()

model_info = get_model_complexity_info(
    model, (3, 224, 224), print_per_layer_stat=True)
print(model_info)

input_res = (3, 224, 224)
input_tensor = torch.randn(input_res)
out = model(input_tensor)
print(out.shape)
