
import numpy as np
import torch
import torchvision

from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


# define the train / validation dataset loader, using the SubsetRandomSampler for the splitt:
class LoadSplitData:
    """split data to create the partitions to train, val and test."""

    def __call__(self, DATA_DIR: str, TEST_SPLIT: float):
        """Split data to create the partitions to train, val and test.

        Parameters
        ----------
        DATA_DIR: str
            path where are the images data to train and val.
        TEST_SPLIT:
           Percentage to split test dataset.
        """
        data_transforms = transforms.Compose(
            [
                transforms.Resize((150, 150)),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

            ]
        )

        data_transforms_overfitting = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, saturation=0.8, hue=0.5),
            transforms.Resize((150, 150)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.ToTensor()
          ])
            #transforms.CenterCrop(5),
            #transforms.RandomGrayscale(),
            #transforms.RandomAutocontrast(),

        dataset = datasets.ImageFolder(
            DATA_DIR, transform=data_transforms, target_transform=lambda x: x
        )

        test_size = int(len(dataset) * TEST_SPLIT)
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [len(dataset) - test_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        val_size = int(len(train_dataset) * TEST_SPLIT)
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset,
            [len(train_dataset) - val_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=4, shuffle=True
        )

        valloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=4, shuffle=False
        )
        testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=4, shuffle=False
        )

        return trainloader, testloader, valloader



if __name__ == "__main__":
    DATA_DIR = "/Users/andreiapfsousa/projects_andreiapfsousa/ComputerVisionProjects/Breeds_project/Dog_images"
    #TEST_DIR = "/Users/andreiapfsousa/projects_andreiapfsousa/ComputerVisionProjects/videos_pilar/test"
    t = LoadSplitData()
    TEST_SPLIT=0.3
    trainloader, testloader,  valloader = t(DATA_DIR, TEST_SPLIT)

    for sample, label in testloader:
        print("labels of testloader:", label)
        print(sample.shape)

    for sample, label in trainloader:
        print("labels of trainloader:", label)
        print(sample.shape)

    for sample, label in valloader:
        print("labels of valloader:", label)
        print(sample.shape)
