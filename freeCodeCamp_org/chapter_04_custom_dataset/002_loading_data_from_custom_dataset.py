import os
import pathlib

import torch

from PIL import Image
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms


# Make function to find classes in target directory
def find_classes(directory: str) -> tuple[list[str], dict[str, int]]:
    """Finds the class folder names in a target directory.

    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        tuple[list[str], dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

    Example:
        find_classes("food_images/train")
        (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: list[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    # 2. Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")

    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(16, 8))

    # 6. Loop through samples and display random samples 
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image.permute(1, 2, 0)
        # Plot adjusted samples
        plt.subplot(1, n, i + 1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
    plt.show()

        
class ImageFolderCustom(Dataset):
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: [str, pathlib.Path], transform=None) -> None:
        """
        Create class attributes
        :param targ_dir:
        :param transform:
        """
        # Get all image paths
        self.paths = list(
            pathlib.Path(targ_dir).glob("*/*.jpg"))  # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        """
        Opens an image via a path and returns it
        :param index:
        :return:
        """
        return Image.open(self.paths[index])

    def __len__(self) -> int:
        """
        Overwrites the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
        :return:  the total number of samples
        """
        return len(self.paths)

    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        :param index: item index
        :return: ne sample of data, data and label (X, y)
        """
        img = self.load_image(index)
        class_name = self.paths[index].parent.name  # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx  # return data, label (X, y)
        else:
            transform = transforms.Compose([transforms.ToTensor()])
            return transform(img), class_idx  # return data, label (X, y)


data_path = pathlib.Path("data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / 'train'
test_dir = image_path / 'test'

# Augment train data
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_data_custom = ImageFolderCustom(targ_dir=train_dir, transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir, transform=test_transforms)
print(train_data_custom, test_data_custom)
print(len(train_data_custom), len(test_data_custom))
print(train_data_custom.classes)
print(train_data_custom.class_to_idx)

display_random_images(train_data_custom,
                      n=12,
                      classes=train_data_custom.classes,
                      seed=None)

train_dataloader_custom = DataLoader(dataset=train_data_custom,  # use custom created train Dataset
                                     batch_size=1,  # how many samples per batch?
                                     num_workers=0,  # how many subprocesses to use for data loading? (higher = more)
                                     shuffle=True)  # shuffle the data?

test_dataloader_custom = DataLoader(dataset=test_data_custom, # use custom created test Dataset
                                    batch_size=1,
                                    num_workers=0,
                                    shuffle=False)  # don't usually need to shuffle testing data

print(train_dataloader_custom, test_dataloader_custom)
img_custom, label_custom = next(iter(train_dataloader_custom))
# Batch size will now be 1, try changing the batch_size parameter above and see what happens
print(f"Image shape: {img_custom.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label_custom.shape}")
