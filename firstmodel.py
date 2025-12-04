import torch
import torch.nn as nn

class SimlpeNN(nn.Module):
    def __init(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)
    
model = SimlpeNN()
print(model)