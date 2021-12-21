import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import model
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython import embed


from breeds_project.pre_processing.transform_split_data import LoadSplitData
from breeds_project.model.model import SqueezeNet

parser = argparse.ArgumentParser('Options for training SqueezeNet in pytorch')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size of train')
parser.add_argument('--epoch', type=int, default=50, metavar='N', help='number of epochs to train for')
parser.add_argument('--learning-rate', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--no-cuda', action='store_true', default=False, help='use cuda for training')
parser.add_argument('--log-schedule', type=int, default=10, metavar='N', help='number of epochs to save snapshot after')
parser.add_argument('--seed', type=int, default=42, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--model_name', type=str, default=None, help='Use a pretrained model')
parser.add_argument('--want_to_test', type=bool, default=True, help='make true if you just want to test')
parser.add_argument('--epoch_50', action='store_true', help='would you like to use 55 epoch learning rule')
parser.add_argument('--num_classes', type=int, default=120, help="how many classes training for")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# import and split data:
DATA_DIR = "/Users/andreiapfsousa/projects_andreiapfsousa/ComputerVisionProjects/Breeds_project/Dog_images"
num_classes = 120
TEST_SPLIT = 0.3
t = LoadSplitData()
trainloader, testloader, valloader = t(DATA_DIR, TEST_SPLIT)

# get the model and convert it into cuda for if necessary
net = SqueezeNet()
if args.model_name is not None:
    print("loading pre trained weights")
    pretrained_weights = torch.load(args.model_name)
    net.load_state_dict(pretrained_weights)

if args.cuda:
    net.cuda()
print(net)


# create optimizer
# using the 55 epoch learning rule here
def paramsforepoch(epoch):
    p = dict()
    regimes = [[1, 18, 5e-3, 5e-4],
               [19, 29, 1e-3, 5e-4],
               [30, 43, 5e-4, 5e-4],
               [44, 52, 1e-4, 0],
               [53, 1e8, 1e-5, 0]]
    # regimes = [[1, 18, 1e-4, 5e-4],
    #            [19, 29, 5e-5, 5e-4],
    #            [30, 43, 1e-5, 5e-4],
    #            [44, 52, 5e-6, 0],
    #            [53, 1e8, 1e-6, 0]]
    for i, row in enumerate(regimes):
        if epoch >= row[0] and epoch <= row[1]:
            p['learning_rate'] = row[2]
            p['weight_decay'] = row[3]
    return p


avg_loss = list()
best_accuracy = 0.0
fig1, ax1 = plt.subplots()


# train the model
# TODO: Compute training accuracy and test accuracy

# create a temporary optimizer
optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)


def adjustlrwd(params):
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = params['learning_rate']
        param_group['weight_decay'] = params['weight_decay']


# train the network
def train(epoch):

    # set the optimizer for this epoch
    if args.epoch_55:
        params = paramsforepoch(epoch)
        print("Configuring optimizer with lr={:.5f} and weight_decay={:.4f}".format(params['learning_rate'], params['weight_decay']))
        adjustlrwd(params)
    ###########################################################################

    global avg_loss
    correct = 0
    net.train()
    for b_idx, (data, targets) in enumerate(trainloader):

        if args.cuda:
            data, targets = data.cuda(), targets.cuda()
        # convert the data and targets into Variable and cuda form
        data, targets = Variable(data), Variable(targets)

        # train the network
        optimizer.zero_grad()
        scores = net.forward(data)
        scores = scores.view(args.batch_size, args.num_classes)
        loss = F.nll_loss(scores, targets)

        # compute the accuracy
        pred = scores.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(targets.data).cpu().sum()

        avg_loss.append(loss.data[0])
        loss.backward()
        optimizer.step()

        if b_idx % args.log_schedule == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (b_idx+1) * len(data), len(trainloader.dataset),
                100. * (b_idx+1)*len(data) / len(trainloader.dataset), loss.data[0]))

            # also plot the loss, it should go down exponentially at some point
            ax1.plot(avg_loss)
            fig1.savefig("Squeezenet_loss.jpg")

    # now that the epoch is completed plot the accuracy
    train_accuracy = correct / float(len(trainloader.dataset))
    print("training accuracy ({:.2f}%)".format(100*train_accuracy))
    return (train_accuracy*100.0)


def val():
    global best_accuracy
    correct = 0
    net.eval()
    for idx, (data, target) in enumerate(valloader):
        if idx == 73:
            break

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        # do the forward pass
        score = net.forward(data)
        pred = score.data.max(1)[1] # got the indices of the maximum, match them
        correct += pred.eq(target.data).cpu().sum()

    print("predicted {} out of {}".format(correct, 73*64))
    val_accuracy = correct / (73.0*64.0) * 100
    print("accuracy = {:.2f}".format(val_accuracy))

    # now save the model if it has better accuracy than the best model seen so forward
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        # save the model
        torch.save(net.state_dict(), 'bsqueezenet_onfulldata.pth')
    return val_accuracy


def test():
    # load the best saved model
    weights = torch.load('bsqueezenet_onfulldata.pth')
    net.load_state_dict(weights)
    net.eval()

    test_correct = 0
    total_examples = 0
    accuracy = 0.0
    for idx, (data, target) in enumerate(testloader):
        if idx < 73:
            continue
        total_examples += len(target)
        data, target = Variable(data), Variable(target)
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        scores = net(data)
        pred = scores.data.max(1)[1]
        test_correct += pred.eq(target.data).cpu().sum()
    print("Predicted {} out of {} correctly".format(test_correct, total_examples))
    return 100.0 * test_correct / (float(total_examples))


if __name__ == '__main__':
    if not args.want_to_test:
        fig2, ax2 = plt.subplots()
        train_acc, val_acc = list(), list()
        for i in range(1, args.epoch+1):
            train_acc.append(train(i))
            val_acc.append(val())
            ax2.plot(train_acc, 'g')
            ax2.plot(val_acc, 'b')
            fig2.savefig('train_val_accuracy.jpg')
    else:
        test_acc = test()
        print("Testing accuracy on CIFAR-10 data is {:.2f}%".format(test_acc))