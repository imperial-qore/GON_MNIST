import torch
import torch.nn as nn
import torch.nn.functional as F

class mnist(nn.Module):
    def __init__(self):
        super(mnist, self).__init__()
        self.name = "mnist"
        self.activation = F.relu
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 256)
        self.fc2 = nn.Linear(256, 10)
        self.output = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = x.flatten()
        x = self.activation(self.fc1(x))
        x = self.output(self.fc2(x))
        return x

