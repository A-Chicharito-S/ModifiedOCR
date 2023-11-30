import torch

a = torch.ones(5)

a[0] = 2
print(a)

a = a /a.max()
print(a)