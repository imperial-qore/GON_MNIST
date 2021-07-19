import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import *

class mnist(nn.Module):
    def __init__(self):
        super(mnist, self).__init__()
        self.name = "mnist"
        self.activation = nn.LeakyReLU(True)
        self.conv1 = nn.Conv2d(1, 128, 5, 1, 2)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 16, 3, 1, 1)
        self.conv5 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv6 = nn.Conv2d(16, 16, 3, 1, 1)
        self.fc1 = nn.Linear(12544, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2*N_CLASSES)
        self.output = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x))
        x = x.flatten()
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.output(self.fc4(x))
        return x

class fashionmnist(nn.Module):
    def __init__(self):
        super(fashionmnist, self).__init__()
        self.name = "fashionmnist"
        self.activation = nn.LeakyReLU(True)
        self.conv1 = nn.Conv2d(1, 128, 5, 1, 2)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 16, 3, 1, 1)
        self.conv5 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv6 = nn.Conv2d(16, 16, 3, 1, 1)
        self.fc1 = nn.Linear(12544, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2*N_CLASSES)
        self.output = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x))
        x = x.flatten()
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.output(self.fc4(x))
        return x


class mnist2(nn.Module):
    def __init__(self):
        super(mnist2, self).__init__()
        self.name = "mnist2"
        self.find = nn.Sequential(nn.Linear(784, 512),
                      nn.Tanh(),
                      nn.Linear(512, 256),
                      nn.Tanh(),
                      nn.Linear(256, 128),
                      nn.Tanh(),
                      nn.Linear(128, 128),
                      nn.Tanh(),
                      nn.Linear(128, 2*N_CLASSES),
                      nn.Softmax(dim=0))

    def forward(self, x):
        x = self.find(x.flatten())
        return x


