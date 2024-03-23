import os.path
import pprint

import torch
from torchvision import datasets, transforms, models
from torchvision.transforms import v2
from torch.utils.data import Dataset
from torchvision.io import read_image
import warnings

warnings.filterwarnings('ignore')


class Food101PartialLabels(Dataset):
    def __init__(self, root_folder: str, split: str, labels_indices: list, transform=None):
        self.root_folder = root_folder
        self.labels = None
        self.classes = None
        self.indices = None
        self.transform = transform
        self.__split = split
        self.__images_number = -1
        self.__number_source_classes = 101
        self.__images_extensions = '.jpg'
        self.__number_classes = -1
        self.__images_labels: list
        self.__images_filenames = []
        self.__train_samples_range = (0, 750)
        self.__test_samples_range = (750, 1000)
        self.__images_filepaths = []
        self.__images_label_indices = []
        self.process_data(labels_indices)

    def __len__(self):
        return self.__images_number

    def __getitem__(self, index):
        image_path = os.path.join(os.path.dirname(__file__),
                                  self.root_folder,
                                  self.__images_filepaths[index])
        image = read_image(str(image_path))
        label = self.__images_label_indices[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def process_data(self, labels_indices: list):
        if not len(labels_indices):
            raise ValueError('label_indices can not be empty')
        if not all(isinstance(x, int) for x in labels_indices):
            raise ValueError('label_indices should contain only integer values')
        self.__number_classes = len(labels_indices)
        if not all(0 <= x < self.__number_classes for x in labels_indices):
            raise ValueError('label_indices should be <= 100')

        sub_folder_meta = 'food-101/meta'
        sub_folder_images = 'food-101/images'
        if os.path.exists(self.root_folder):
            data_meta_path = os.path.join(str(os.path.dirname(__file__)),
                                          self.root_folder,
                                          sub_folder_meta)
            data_images_path = os.path.join(str(os.path.dirname(__file__)),
                                            self.root_folder,
                                            sub_folder_images)
        else:
            raise OSError('Error loading data from', self.root_folder)

        classes_filepath = os.path.join(data_meta_path, 'classes.txt')
        labels_filepath = os.path.join(data_meta_path, 'labels.txt')
        with open(classes_filepath) as file_classes, open(labels_filepath) as file_labels:
            all_classes = file_classes.readlines()
            all_labels = file_labels.readlines()

        self.indices = [i for i in range(self.__number_classes)]
        self.classes = [all_classes[i].replace('\n', '') for i in labels_indices]
        self.labels = [all_labels[i].replace('\n', '') for i in labels_indices]

        self.__images_number = 0
        for index, folder in enumerate(self.classes):
            label_images_path = os.path.join(os.path.dirname(__file__), data_images_path, folder)
            label_filenames = [f for f in os.listdir(label_images_path) if f.endswith(self.__images_extensions)]

            if self.__split == 'train':
                label_filenames = label_filenames[self.__train_samples_range[0]:self.__train_samples_range[1]]
            elif self.__split == 'test':
                label_filenames = label_filenames[self.__test_samples_range[0]:self.__test_samples_range[1]]

            folder_images_number = len(label_filenames)
            self.__images_number += folder_images_number
            label_filenames = [os.path.join(folder, label) for label in label_filenames]
            label_indices = [index] * folder_images_number
            self.__images_filepaths.extend(label_filenames)
            self.__images_label_indices.extend(label_indices)


class Food101PartialData(Dataset):
    def __init__(self, root_folder, labels_indices):
        self.root_folder = root_folder
        self.labels_indices = labels_indices

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


root_path = 'dataset_food101'
batch_size = 16
number_workers = 12

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # v2.RandomCrop(size=(224, 224)),
    # v2.RandomPhotometricDistort(contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.02, 0.02), p=0.5),
    v2.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1)),
    # transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_loader_full_train = torch.utils.data.DataLoader(
    datasets.Food101(root_path, 'train', transform, download=True),
    batch_size=batch_size,
    shuffle=True,
    num_workers=number_workers,
)
data_loader_full_test = torch.utils.data.DataLoader(
    datasets.Food101(root_path, 'test', transform, download=True),
    batch_size=batch_size,
    shuffle=True,
    num_workers=number_workers,
)

number_classes = len(data_loader_full_train.dataset.classes)

print('Train full Food101 dataset length:', len(data_loader_full_train), 'Test  dataset length:',
      len(data_loader_full_test))
print('Full Food101 dataset classes number:', number_classes)
print('Full Food101 dataset classes:')
pprint.pprint(data_loader_full_train.dataset.classes)

dataset_partial_labels = Food101PartialLabels(
    root_folder=root_path,
    labels_indices=[0, 1, 2, 3, 4],
    split='train'
)

# dataset_partial_data = Food101PartialData(
#
# )
