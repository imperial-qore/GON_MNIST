import os
from src.constants import *
import numpy as np
import random
import numpy as np
import torch
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

def filter(dset, c):
	dset2 = []
	for a, b in dset: 
		if b == c: dset2.append((a,b))
	return dset2

def load_mnist_data():
	trainset = datasets.MNIST(DATASET_SAVE_PATH, download=True, train=True, transform=transform)
	valset = datasets.MNIST(DATASET_SAVE_PATH, download=True, train=False, transform=transform)
	if N_CLASSES == 1:
		return filter(trainset, 0), filter(valset, 0)
	return trainset, valset

def load_mnist2_data():
	return load_mnist_data()

def freeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = False

def unfreeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = True

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

def plot_images(fake, real, iteration):
	w, h = len(fake), 2
	fig = plt.figure(figsize=(w, h))
	for i in range(1, w*h +1):
	    img = fake[i - 1] if i <= w else real[i - w - 1]
	    img = img[0].data.view(1,28,28).numpy().squeeze()
	    ax = fig.add_subplot(h, w, i); ax.set_xticks([]); ax.set_yticks([])
	    plt.imshow(img, cmap='gray_r')
	plt.savefig('data_'+str(iteration)+'.png')
	plt.close()