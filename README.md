# ComputerVisionProjects

Computer Vision project:

- On ambit of computer vision class the idea is to create a model to classify if is a beagle or not ( binary classification) .

## Structure

The main goal of this project is to understand better and apply the tech- niques/methods and the knowledge I learned during the Computer Vision classes. My idea was to use computer vision to solve a real problem related to distin- guishing dog breeds. The objective is to classify a dog in one image as a beagle or other breed.


So, the steps to develop this project are:
    - Collect data and create a balanced dataset;
    - Define a pre-trained model and other simple neural network model;
    - Define hyperparameters and the data augmentation pipeline;
    - Run different experiments with different models and apply the data augmentation to improve the robustness and generalization of the model;
    - Analyse and compare different results when I use data augmentation vs without data augmentation and pre-trained neural network vs a simple neural network (from scratch).
To develop the code of this project, I used a tutorial from PyTorch: PyTorch and Albumentations for image classification.


## Input data generation

In total I have 2 003 images: Beagle: 1 001 images; Other breeds: 1 002 images.


## Experiments
On the first experiment(without data augmentation) I used the data of beagles vs other breeds without big transformations on data to understand how the models comport. The dataset is a balanced dataset and I have different images with different backgrounds and the dogs are on different positions and angles.
In the other experiments, I applied different combinations of data augmen- tation methods and analyze the behaviour of the model and results.
Also, I compared the results for the pre-trained model vs a simple Neural Network (from scratch).
