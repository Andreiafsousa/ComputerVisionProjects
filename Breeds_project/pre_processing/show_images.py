import matplotlib.pyplot as plt
import numpy as np
import torchvision

from torchvision import datasets
import torchvision.transforms as transforms

from breeds_project.pre_processing import LoadSplitData


# see the images:
def imshow(self, img):
    """To show the images."""
    img = img/2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


DATA_DIR = "/Users/andreiapfsousa/projects_andreiapfsousa/ComputerVisionProjects/Breeds_project/Dog_images"
data_transforms = transforms.Compose(
    [
        transforms.Resize((150, 150)),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ]
)
dataset = datasets.ImageFolder(
    DATA_DIR, transform=data_transforms, target_transform=lambda x: x
)

#classes = dataset.class_to_idx

t = LoadSplitData()
TEST_SPLIT = 0.3
trainloader, testloader,  valloader = t(DATA_DIR, TEST_SPLIT)


# get some random training images
dataiter = iter(dataset)
images, labels = dataiter.next()

# show images
t.imshow(torchvision.utils.make_grid(images))
# print labels
# print(" ".join("%5s" % classes[labels[j]] for j in range(4)))
