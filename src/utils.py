import os
from src.constants import *
import numpy as np
import random
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = True
plt.rcParams['figure.figsize'] = 2, 2

if not os.path.exists(MODEL_SAVE_PATH):
	os.mkdir(MODEL_SAVE_PATH)

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_one_hot(index, size):
	a = torch.FloatTensor(index.shape[0], size)
	a.zero_()
	y = index.view(-1,1).long()
	a.scatter_(1, y, 1)
	return a

def load_mnist_data():
	trainset = datasets.MNIST(DATASET_SAVE_PATH, download=True, train=True, transform=transform)
	valset = datasets.MNIST(DATASET_SAVE_PATH, download=True, train=False, transform=transform)
	return trainset, valset

def plot_accuracies(accuracy_list):
	trainAcc = [i[0] for i in accuracy_list]
	testAcc = [i[1] for i in accuracy_list]
	plt.xlabel('Epochs')
	plt.ylabel('Average Training Loss')
	plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
	plt.legend(loc=1)
	plt.savefig('training-graph.pdf')
	plt.clf()
	plt.xlabel('Epochs')
	plt.ylabel('Average Testing Loss')
	plt.errorbar(range(len(testAcc)), testAcc, label='Average Testing Loss', alpha = 0.7,\
	    linewidth = 1, linestyle='dotted', marker='+')
	plt.legend(loc=4)
	plt.savefig('testing-graph.pdf')
