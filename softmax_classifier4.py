import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

best_accuracy = 0.0
checkpoint_dir = '../checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

best_model_path = os.path.join(checkpoint_dir, 'best_model.ckpt')

def train(epoch):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch}')
    for batch_idx, (inputs, target) in progress_bar:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({'Loss': running_loss / (batch_idx + 1)})

def test(epoch):
    global best_accuracy, best_model_path
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100.0 * correct / total
        print(f'Accuracy on test set: {accuracy:.2f}%')

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Remove old best model
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            # Save new best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy
            }, best_model_path)
            print(f'New best model saved with accuracy: {accuracy:.2f}%')

            # Rename best model
            rename_model(accuracy)

def rename_model(accuracy):
    old_name = best_model_path
    new_name = os.path.join(checkpoint_dir, f'mnist_classifier_best_model_{accuracy:.2f}.ckpt')
    if os.path.exists(new_name):
        os.remove(new_name)
    os.rename(old_name, new_name)
    print(f'Model renamed to: {new_name}')

if __name__ == '__main__':
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(epoch)
        test(epoch)
