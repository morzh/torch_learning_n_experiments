import torch

A = torch.rand(3, 3)
B = torch.rand(3, 3)
print(A == B)

torch.manual_seed(42)
C = torch.rand(3, 3)
torch.manual_seed(42)
D = torch.rand(3, 3)
print(C == D)
