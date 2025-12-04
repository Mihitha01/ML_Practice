import torch #torch → main PyTorch library (tensors, GPU operations).
import torch.nn as nn #torch.nn → contains classes to build neural networks (layers, activations).
import torch.optim as optim #torch.optim → contains optimization algorithms (SGD, Adam, etc.)

"""PyTorch is designed like Lego blocks:
    nn → build the model
    optim → train the model
    tensor → data structure
"""
# 1) Sample training data
x = torch.randn(100, 3)   # 100 samples, 3 features
y = torch.randn(100, 1)   # 100 labels, 1 output

# 2) Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(3, 5)  # input 3 → hidden 5
        self.layer2 = nn.Linear(5, 1)  # hidden 5 → output 1

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 3) Create model
model = SimpleNN()

# 4) Loss function + optimizer
criterion = nn.MSELoss()                 # Mean squared error
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 5) Training loop
for epoch in range(100):
    optimizer.zero_grad()               # reset gradients
    output = model(x)                   # forward pass
    loss = criterion(output, y)         # compute loss
    loss.backward()                     # backpropagation
    optimizer.step()                    # update weights

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Final Loss:", loss.item())


