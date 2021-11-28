import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from torchvision import datasets
import torchvision.transforms as transforms


# define the train / validation dataset loader, using the SubsetRandomSampler for the split:
DATA_DIR = "/Users/andreiapfsousa/projects_andreiapfsousa/ComputerVisionProjects/videos_pilar/train"


class LoadSplitData:
    """split data to create the partitions to train, val and test."""

    def __call__(self, DATA_DIR: str, TEST_SPLIT: float):
        """Split data to create the partitions to train, val and test.

        Parameters
        ----------
        DATA_DIR: str
            path where are the images data to train.
        TEST_SPLIT:
            percentage to split the dataset.
        """
        data_transforms = transforms.Compose(
            [
                transforms.Resize((50, 50)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

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
            val_dataset, batch_size=4, shuffle=True
        )
        testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=4, shuffle=True
        )

        return trainloader, testloader, valloader

    def imshow(self, img):
        """To show the images."""
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


if __name__ == "__main__":
    DATA_DIR = "/Users/andreiapfsousa/projects_andreiapfsousa/ComputerVisionProjects/videos_pilar/train"
    TEST_SPLIT = 0.2
    t = LoadSplitData()
    trainloader, testloader, valloader = t(DATA_DIR, TEST_SPLIT)

    for sample, label in testloader:
        print("labels of testloader:", label)
        print(sample.shape)

    for sample, label in trainloader:
        print("labels of trainloader:", label)

    for sample, label in valloader:
        print("labels of valloader:", label)

    # see the images:
    classes = ("deitada", "sentada")
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    t.imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join("%5s" % classes[labels[j]] for j in range(4)))
