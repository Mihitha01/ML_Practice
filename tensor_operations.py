import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a + b)

print(a * b)

print(a / b)

x = torch.randn(2, 3)
print(x.shape)

y = x.view(3, 2)
print(y.shape)

