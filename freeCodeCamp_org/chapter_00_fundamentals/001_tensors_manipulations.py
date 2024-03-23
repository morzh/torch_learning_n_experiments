import torch


sample_tensor = torch.rand(3,2)
print(sample_tensor, sample_tensor.shape, sample_tensor.size())
print('-' * 70)
sample_tensor_unsqueezed = sample_tensor.unsqueeze(dim=0)
print(sample_tensor_unsqueezed)

