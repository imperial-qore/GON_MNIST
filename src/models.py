import torch
import torch.nn as nn
import torch.nn.functional as F

class mnist(nn.Module):
    def __init__(self):
        super(mnist, self).__init__()
        self.name = "mnist"
        self.activation = nn.Tanh()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 256)
        self.fc2 = nn.Linear(256, 11)
        self.output = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = x.flatten()
        x = self.activation(self.fc1(x))
        x = self.output(self.fc2(x))
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
                      nn.Linear(128, 11),
                      nn.Softmax(dim=0))

    def forward(self, x):
        x = self.find(x.flatten())
        return x


