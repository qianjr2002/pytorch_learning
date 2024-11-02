import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from ptflops import get_model_complexity_info

batch_size = 64

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

def train(model, device, train_dataloader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, target) in enumerate(train_dataloader, 0):
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss : %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

def test(model, device, test_dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, target in test_dataloader:
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    model = Net().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     train(model, device, train_dataloader, optimizer, criterion, epoch)
    #     test(model, device, test_dataloader)
    # Calculate MACs and FLOPs
    macs, params = get_model_complexity_info(model, (1, 28, 28), as_strings=True, print_per_layer_stat=True)
    print(f"MACs: {macs}")
    print(f"FLOPs: {params}")
if __name__ == '__main__':
    main()
