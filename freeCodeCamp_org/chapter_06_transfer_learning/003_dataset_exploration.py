import os
import pprint
from collections import Counter
import pathlib

from PIL import Image
from PIL.ExifTags import TAGS

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
from glob import glob

import importlib
dataset = importlib.import_module('002_data_loaders', 'dataset')
# dataset = __import__('002_data_loaders')

root_path = 'dataset_food101/food-101/images'

explore_classes_distribution = False
explore_dimensions_distribution = True
explore_brightness_distribution = True

if explore_classes_distribution:
    print('Calculating classes distribution. This will take some time')
    targets = [item for (_, labels) in dataset.data_train for item in labels.tolist()]
    idx_to_class = {v: k for k, v in dataset.data_train.dataset.class_to_idx.items()}
    idx_distribution = dict(Counter(targets))
    classes_distribution = {idx_to_class[k]: v for k, v in idx_distribution.items()}
    pprint.pprint(classes_distribution)

    plt.figure(figsize=(10, 5))
    sb.countplot(classes_distribution)
    plt.show()


if explore_dimensions_distribution:
    images_files_paths = list(pathlib.Path(root_path).glob("*/*.jpg"))
    dataset_image_dimensions = np.empty((3, 0))
    for image_file in images_files_paths:
        current_image = Image.open(image_file.parent / image_file.name)
        current_dimensions_data = np.array([current_image.width,
                                            current_image.height,
                                            current_image.width / current_image.height]).reshape(3, 1)
        dataset_image_dimensions = np.hstack((dataset_image_dimensions, current_dimensions_data))

    plt.figure(figsize=(30, 20))
    plt.subplot(131)
    plt.title('Images widths distribution')
    plt.hist(dataset_image_dimensions[0], bins=15)
    plt.xlabel('images widths')
    plt.ylabel('images number')
    plt.subplot(132)
    plt.title('Images heights distribution')
    plt.hist(dataset_image_dimensions[1], bins=15)
    plt.xlabel('images heights')
    plt.ylabel('images number')
    plt.subplot(133)
    plt.title('Images aspect ratio distribution')
    plt.hist(dataset_image_dimensions[2], bins=15)
    plt.xlabel('aspect ratio')
    plt.ylabel('images number')
    plt.tight_layout()
    plt.show()


if explore_brightness_distribution:
    pass
