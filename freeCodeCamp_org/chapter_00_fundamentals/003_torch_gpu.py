import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

tensor = torch.tensor([1, 2, 3])
print(tensor, tensor.device)

tensor_gpu = tensor.to(device)
print(tensor_gpu, tensor_gpu.device)

tensor_gpu_numpy = tensor_gpu.cpu().numpy()
print(tensor_gpu_numpy)

