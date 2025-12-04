import torch #torch → main PyTorch library (tensors, GPU operations).
import torch.nn as nn #torch.nn → contains classes to build neural networks (layers, activations).
import torch.optim as optim #torch.optim → contains optimization algorithms (SGD, Adam, etc.)

"""
PyTorch is designed like Lego blocks:
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

"""
Layer 1
    Takes 3 inputs (because x has 3 features)
    Produces 5 outputs
    This is called a hidden layer
    PyTorch creates:
        a weight matrix of size [5 × 3]
        a bias vector of size [5]

Layer 2
    Takes 5 inputs (output of layer1)
    Produces 1 output (prediction)
    PyTorch creates:
        weight matrix [1 × 5]
        bias [1] 

Total Learnable Parameters
    You have (5×3 + 5) + (1×5 + 1) = 26 parameters         
"""

"""
    The forward() function defines how data flows through your network.

    STEP1: self.layer1(x)
        This performs : output = x * weights + bias
        This is the core mathematical operation of neural networks.

    STEP2: torch.relu()
        ReLU = Rectified Linear Unit
        It makes the model nonlinear: if x > 0 → x
                                      if x ≤ 0 → 0
        This helps the network learn complex patterns.

    STEP3: self.layer2(x)
        Another linear transformation to produce final output.

"""

# 3) Create model
model = SimpleNN()

"""
PyTorch: builds the network 
         initializes random weights and biases
         prepares memory for training
"""

# 4) Loss function + optimizer
criterion = nn.MSELoss()  #Mean Squared Error = measures how far predictions are from labels.
optimizer = optim.SGD(model.parameters(), lr=0.01)

"""
If loss → small ⇒ model is learning
If loss → large ⇒ model is bad

Explanation:
    SGD = Stochastic Gradient Descent
    Takes all model parameters (weights+biases)
    lr=0.01 = learning rate
        → controls how big the weight updates are
The optimizer is responsible for learning.
"""

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

"""
Explanation:

Training happens in "epochs" (full passes over data).
Here you train for 100 epochs.

Step 1 — Reset Gradients:
    optimizer.zero_grad() -> PyTorch accumulates gradients by default.
    If you don’t reset: grad(new) = grad(old) + grad(current batch)
                        This causes exploding gradients.
                        So you must clear them every loop.

Step 2 — Forward Pass:
    output = model(x) This internally calls: layer1 → relu → layer2
                                             You get the model prediction.

Step 3 — Compute Loss:
    loss = criterion(output, y) -> This computes how wrong the model is.    

Step 4 — Backpropagation:
    This is the most important part of training.

    PyTorch:
        computes partial derivatives
        calculates how much each weight contributed to the loss
        stores gradients inside each parameter
        This process is called backpropagation.

Step 5 — Update Weights
    optimizer.step() -> This uses the gradients to update weights:
                            w = w - learning_rate * gradient

                        This is how the model "learns".                                              
"""     
"""
🧠 What Your Model is Actually Doing?
Your neural network is:
    1.Taking 3-features input
    2.Passing through a hidden layer with ReLU
    3.Producing a single output
    4.Comparing output to target
    5.Calculating error
    6.Adjusting weights using gradients
    7.Repeating 100 times
    8.Slowly reducing the loss
Even with random data, the model tries to minimize the difference by adjusting weights.

"""


