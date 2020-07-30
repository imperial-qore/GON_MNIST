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
	a = np.zeros(size)
	a[index] = 1
	return torch.FloatTensor(a)

def load_mnist_data():
	trainset = datasets.MNIST(DATASET_SAVE_PATH, download=True, train=True, transform=transform)
	valset = datasets.MNIST(DATASET_SAVE_PATH, download=True, train=False, transform=transform)
	return trainset, valset

def load_mnist2_data():
	return load_mnist_data()

def plot_accuracies(accuracy_list):
	trainAcc = [i[1] for i in accuracy_list]
	testAcc = [i[0] for i in accuracy_list]
	plt.xlabel('Epochs')
	plt.ylabel('Average Training Loss')
	plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
	plt.legend(loc=1)
	plt.savefig('training-graph.pdf')
	plt.clf()
	plt.xlabel('Epochs')
	plt.ylabel('Average Testing Accuracy')
	plt.errorbar(range(len(testAcc)), testAcc, label='Average Testing Accuracy', alpha = 0.7,\
	    linewidth = 1, linestyle='dotted', marker='+')
	plt.legend(loc=4)
	plt.savefig('testing-graph.pdf')

def plot_image(data, iteration):
	plt.imsave('test_'+str(iteration)+'.png', data, cmap='gray_r')
