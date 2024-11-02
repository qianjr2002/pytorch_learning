import torch

# y = 2*x**2+3*x+1
x_data = [0.0, 1.0, 2.0, 3.0, 4.0]
y_data = [1.0, 6.0, 15.0, 28.0, 45.0]

w1 = torch.tensor([1.5], requires_grad=True)
w2 = torch.tensor([2.5], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

def forward(x):
    return w1 * x**2 + w2 * x + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2

print('Predict before training', 5, forward(5).item())

learning_rate = 0.001

for epoch in range(1000):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            b -= learning_rate * b.grad
        
        w1.grad.zero_()
        w2.grad.zero_()
        b.grad.zero_()
    
    if epoch % 10 == 0 or epoch == 199:
        print(f"Epoch {epoch+1}: Loss = {l.item():.6f}")
        if torch.isnan(l) or torch.isinf(l):
            print("Encountered NaN or Inf loss value. Stopping training.")
            break

print('Predict after training', 5, forward(5).item())
print(f"Final parameters: w1 = {w1.item()}, w2 = {w2.item()}, b = {b.item()}")
