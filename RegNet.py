from __future__ import print_function
import numpy as np
from numpy import genfromtxt
import pandas as pd
import random
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torch.utils.data as data
import torch.nn.init as init
from torchvision import datasets, transforms, utils
import torchvision.models as models
from collections import OrderedDict
from torchsummary import summary

# flush GPU memory
torch.cuda.empty_cache()
# select training/inference device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------------------------------------------------
#                     dataset config and normalization
#-------------------------------------------------------------------------------

# load numerical .csv dataset into a numpy array
dataset = genfromtxt('C:/Users/quentin.munch/Desktop/data.csv', delimiter=';')
# split data to get training and testing data
train_input = dataset[:200000, :3]
train_target = dataset[:200000, 3]
test_input = dataset[200000:250000, :3]
test_target = dataset[200000:250000, 3]


# load data as a tensor variable
train_input = torch.from_numpy(train_input).float()
train_input = Variable(train_input)
train_target = torch.from_numpy(train_target).float()
test_input = torch.from_numpy(test_input).float()
test_input = Variable(test_input)
test_target = torch.from_numpy(test_target).float()
test_target = Variable(test_target)

# dataset generation
training_dataset = data.TensorDataset(train_input, train_target)
train_loader = data.DataLoader(dataset=training_dataset,batch_size=256,shuffle=True)
testing_dataset = data.TensorDataset(test_input, test_target)
test_loader = data.DataLoader(dataset=testing_dataset,batch_size=256,shuffle=True)

#-------------------------------------------------------------------------------
#                       Regression network definition
#-------------------------------------------------------------------------------

class RegNet(torch.nn.Module):
    def __init__(self):
        super(RegNet, self).__init__()
        self.HIDDEN_1 = nn.Linear(3, 50)
        self.DRP_1 = nn.Dropout(p=0.25)
        self.HIDDEN_2 = nn.Linear(50, 50)
        self.DRP_2 = nn.Dropout(p=0.25)
        self.OUT = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.HIDDEN_1(x))
        x = self.DRP_1(x)
        x = F.relu(self.HIDDEN_2(x))
        x = self.DRP_2(x)
        x = self.OUT(x)
        return x

# network instanciation
VDNet = RegNet().to(device)
# print net
print("ResNet50 STRUCTURE : \n", VDNet)
summary(VDNet, input_size=(1,3))
print("neural network ready !")

#-------------------------------------------------------------------------------
#                           Optimizer config
#-------------------------------------------------------------------------------

# SGD optimizer with Nesterow momentum
optimizer = optim.SGD(VDNet.parameters(), lr = 0.01,
                                            momentum = 0.90,
                                            weight_decay = 1e-6,
                                            nesterov = True)
# cost function
loss_function = torch.nn.MSELoss()
# number of epoch
number_epoch = 50
# Learning rate scheduler (decreasing polynomial)
lrPower = 2
lambda1 = lambda epoch: (1.0 - epoch / number_epoch) ** lrPower
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])

#-------------------------------------------------------------------------------
#                           Learning procedure
#-------------------------------------------------------------------------------


print("start training")
training_loss = []
test_loss = []
for epoch in range(number_epoch):
    print("--------------------> Epoch = ", epoch)
    avgTrainLoss = 0
    for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
        # pass data to the device GPU
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        # predict
        pred = VDNet(input_batch).reshape(-1)
        # learn
        loss = loss_function(pred, target_batch)
        print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        avgTrainLoss+=loss
    avgTrainLoss = avgTrainLoss / 256
    training_loss.append(avgTrainLoss)
    print("testing...")

    with torch.no_grad():
        VDNet.eval()

    avgTestLoss = 0
    for batch_idx, (input_batch_test, target_batch_test) in enumerate(test_loader):
        # pass data to the device GPU
        input_batch_test = input_batch_test.to(device)
        target_batch_test = target_batch_test.to(device)
        # predict
        pred = VDNet(input_batch_test).reshape(-1)
        # evaluate
        loss_eval = loss_function(pred, target_batch_test)
        avgTestLoss+=loss_eval
    avgTestLoss = avgTestLoss / 256
    test_loss.append(avgTestLoss)
    # update scheduler
    scheduler.step()
    torch.cuda.empty_cache()

# plot training loss
# plot validation loss
plt.figure(1)
plt.title("Validation loss/F1 score")
plt.subplot(211)
nbe = list(range(len(training_loss)))
plt.plot(nbe, training_loss ,'r', label='training Loss')
plt.legend()
plt.subplot(212)
nbr = list(range(len(test_loss)))
plt.plot(nbr, test_loss ,'b', label='validation loss')
plt.legend()
plt.show()

# delta error regression
desired = []
actual = []
for batch_idx, (input_batch_test, target_batch_test) in enumerate(test_loader):
    # pass data to the device GPU
    input_batch_test = input_batch_test.to(device)
    target_batch_test = target_batch_test.to(device)
    # predict
    pred = VDNet(input_batch_test).reshape(-1)
    for index, (value1, value2) in enumerate(zip(pred.cpu().data.numpy(), target_batch_test.cpu().data.numpy())):
        actual.append(value1)
        desired.append(value2)
plt.figure(2)
plt.title("Estimation vs. target")
plt.scatter(desired, actual)
plt.plot([0, 1], [0, 1],  label='identity')
plt.ylabel("Estimation")
plt.xlabel("Target")
plt.legend()
plt.show()
# flush GPU memory
torch.cuda.empty_cache()
