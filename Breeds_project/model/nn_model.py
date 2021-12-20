"""Define a Convolutional Neural Network."""
import torch
import os
import re
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchmetrics

# lightning related imports
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# !pip install -q test_tube transformers pytorch-nlp pytorch-lightning==0.9.0

from breeds_project.pre_processing.transform_split_data import LoadSplitData

# sklearn related imports
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize


# import wandb and login
import wandb
#wandb.init(project="Pilar_computer_vision", entity="andreiapfsousa")

# Callbacks:
## Earlystopping
early_stop_callback = EarlyStopping(
    monitor="val_loss", patience=3, verbose=False, mode="min"
)

## Model Checkpoint Callback

MODEL_CKPT_PATH = "model/"
MODEL_CKPT = "model/model-{epoch:02d}-{val_loss:.6f}"

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", filepath=MODEL_CKPT, save_top_k=1, mode="min"
)


## Custom Callback - ImagePredictionLogger:
class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log(
            {
                "examples": [
                    wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                    for x, pred, y in zip(
                        val_imgs[: self.num_samples],
                        preds[: self.num_samples],
                        val_labels[: self.num_samples],
                    )
                ]
            }
        )


class LitModel(pl.LightningModule):
    """Define a Convolutional Neural Network."""

    def __init__(self, input_shape, num_classes, learning_rate=2e-4, weight_decay=0.001):

        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

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
        # self.pool2 = torch.nn.MaxPool2d(2)

        print(input_shape)
        n_sizes = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_sizes, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 4
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        return x

    # will be used during inference
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # test metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=4)
        print("test_acc", acc)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=4)
        f1_score = torchmetrics.functional.f1(preds, y, average="micro")
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('f1_score', f1_score, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch  # inputs, target
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=3)
        f1_score = torchmetrics.functional.f1(preds, y, average="micro")
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        self.log('f1_score', f1_score, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


# init pipeline:
DATA_DIR = "/Users/andreiapfsousa/projects_andreiapfsousa/ComputerVisionProjects/videos_pilar/train"
num_classes = 120
TEST_SPLIT = 0.3
t = LoadSplitData()
trainloader, testloader, valloader = t(DATA_DIR, TEST_SPLIT)

# Samples required by the custom ImagePredictionLogger callback to log image predictions.
val_samples = next(iter(valloader))
val_imgs, val_labels = val_samples[0], val_samples[1]
val_imgs.shape, val_labels.shape

# Init model
model = LitModel([3, 150, 150], num_classes)

# Initialize wandb logger
wandb_logger = WandbLogger(project="Breeds_computer_vision")

# Initialize a trainer
trainer = pl.Trainer(
    max_epochs=10,
    progress_bar_refresh_rate=5,
    gpus=None,
    logger=wandb_logger,
    callbacks=[early_stop_callback, ImagePredictionLogger(val_samples)],
    checkpoint_callback=checkpoint_callback
)

# Train the model
trainer.fit(model, train_dataloader=trainloader, val_dataloaders=valloader)

# Evaluate the model on the held out test set:
trainer.test(model, test_dataloaders=testloader)

# Close wandb run
wandb.finish()

# # save model:
# #run = wandb.init(project="Pilar_computer_vision", job_type="producer")

# artifact = wandb.Artifact("model", type="model")
# artifact.add_dir(MODEL_CKPT_PATH)


# # Load best model:
# model_ckpts = os.listdir(MODEL_CKPT_PATH)
# losses = []
# for model_ckpt in model_ckpts:
#     loss = re.findall("\d+\.\d+", model_ckpt)
#     losses.append(float(loss[0]))

# best_model_index = np.argsort(losses)[0]
# best_model = model_ckpts[best_model_index]
# #find . -name '.DS_Store' -type f -delete
# print(best_model)

# inference_model = LitModel.load_from_checkpoint(MODEL_CKPT_PATH + best_model)


# # Precision recall curve:

# def evaluate(model, loader):
#     y_true = []
#     y_pred = []
#     for imgs, labels in loader:
#         logits = inference_model(imgs)

#         y_true.extend(labels)
#         y_pred.extend(logits.detach().numpy())

#     return np.array(y_true), np.array(y_pred)


# y_true, y_pred = evaluate(model=inference_model, loader=testloader)

# # generate binary correctness labels across classes
# binary_ground_truth = label_binarize(y_true, classes=[0, 1, 2, 3])

# # compute a PR curve with sklearn like you normally would
# precision_micro, recall_micro, _ = precision_recall_curve(
#     binary_ground_truth.ravel(), y_pred.ravel()
# )

# #run = wandb.init(project="Pilar_computer_vision", job_type='evaluate')

# data = [[x, y] for (x, y) in zip(recall_micro, precision_micro)]
# sample_rate = int(len(data)/100)

# table = wandb.Table(columns=["recall_micro", "precision_micro"], data=data[::sample_rate])
# wandb.log({"precision_recall": wandb.plot.line(table,
#                                                "recall_micro",
#                                                "precision_micro",
#                                                stroke=None,
#                                                title="Average Precision")})
