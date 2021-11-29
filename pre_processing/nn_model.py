"""Define a Convolutional Neural Network."""
import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from transform_split_data import LoadSplitData


class LitModel(pl.LightningModule):
    """Define a Convolutional Neural Network."""
    def __init__(self, input_shape, num_classes, learning_rate=2e-4):

        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate

    # First 2D convolutional layer, taking in 3 input channel (image),
    # outputting 6 convolutional features, with a square kernel size of 5
        self.conv1 = nn.Conv2d(3, 6, 5)
    # Second 2D convolutional layer, taking in the 6 input layers,
    # outputting 16 convolutional features, with a square kernel size of 5
        self.conv2 = nn.Conv2d(6, 16, 5)
    # First fully connected layer
        self.fc1 = nn.Linear(16 * 9 * 9, 120)
        self.fc2 = nn.Linear(120, 84)
    # Second fully connected layer that outputs our 2 labels
        self.fc3 = nn.Linear(84, 2)
    # corners detector, smoothing the images
        # self.pool = nn.MaxPool2d(2, 2)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.pool2 = torch.nn.MaxPool2d(2)

        print(input_shape)
        n_sizes = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_sizes, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        # print(input)
        # input = input.unsqueeze(0)
        # print(input.shape)

        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size


    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x

    # will be used during inference
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x

# init our pipeline:
DATA_DIR = "/Users/andreiapfsousa/projects_andreiapfsousa/ComputerVisionProjects/videos_pilar/train"
TEST_SPLIT = 0.2
num_classes = "2"
t = LoadSplitData()
trainloader, testloader, valloader = t(DATA_DIR, TEST_SPLIT)

for sample, __ in testloader:
    size = sample.shape
print(size)

# Init our model
model = LitModel(size, num_classes)

# Initialize a trainer
trainer = pl.Trainer(max_epochs=50,
                     progress_bar_refresh_rate=20,
                     gpus=1)

# Train the model
trainer.fit(model, trainloader)
