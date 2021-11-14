import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


# define the train / validation dataset loader, using the SubsetRandomSampler for the split:

data_dir = "/Users/andreiapfsousa/projects_andreiapfsousa/ComputerVisionProjects/videos_pilar/train"


def load_split_train_test(datadir, valid_size=0.2):
    train_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    )

    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    print(train_data)

    test_data = datasets.ImageFolder(datadir, transform=test_transforms)
    print(test_data)

    num_train = len(train_data)
    print(num_train)

    indices = list(range(num_train))
    print(indices)

    split = int(np.floor(valid_size * num_train))
    print(split)

    np.random.shuffle(indices)