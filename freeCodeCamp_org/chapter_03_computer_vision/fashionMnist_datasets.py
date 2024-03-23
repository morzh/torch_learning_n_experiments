
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2

import matplotlib.pyplot as plt

show_plot = False
seed = 42


image_transforms = transforms.Compose([transforms.ToTensor(),
                    v2.Normalize(mean=[0.5], std=[0.5]),
                    ])

train_data = datasets.FashionMNIST(root='data_FashionMNIST',
                                   train=True,
                                   download=True,
                                   # transform=transforms.ToTensor(),
                                   transform=image_transforms,
                                   target_transform=None)

test_data = datasets.FashionMNIST(root='data_FashionMNIST',
                                  train=False,
                                  download=True,
                                  transform=image_transforms,
                                  target_transform=None)


print(train_data, test_data)
print(train_data.data.shape, len(test_data.data.shape))
print(train_data.classes)
print(train_data.class_to_idx)
# print(train_data.data[0].max())


class_names = train_data.classes

if show_plot:
    torch.manual_seed(seed)
    fig = plt.figure(figsize=(9, 9))
    rows, cols = 4, 4
    for i in range(1, rows * cols + 1):
        random_idx = torch.randint(0, len(train_data), size=[1]).item()
        img, label = train_data[random_idx]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(class_names[label])
        plt.axis(False)
    plt.show()


batch_size = 32

train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_data,
                         batch_size=batch_size,
                         shuffle=False)

print(f'Dataloaders: {train_loader}, {test_loader}')
print(f'Length of train_loader: {len(train_loader)}')
print(f'Length of test_loader: {len(test_loader)}')
