import random 
import torch
import numpy as np
from copy import deepcopy
from src.utils import *
from src.constants import *
from tqdm import tqdm
import torch.nn.functional as F

import matplotlib.pyplot as plt

def scale(data):
    return torch.max(torch.tensor(-1), torch.min(data, torch.tensor(1)))

def gen(model, data_type, trainloader, num_examples, label, notstart, epsilon=1e-4):
    lr = 0.05
    iteration = 0; equal = 0
    diffs, data, labels = [], [], []
    l, label_vec = torch.nn.CrossEntropyLoss(), torch.LongTensor([label])
    for restart in range(num_examples):
        init = torch.rand((1,1,28,28), dtype=torch.float)*2 - 1
        if not notstart:
            data.append(init); labels.append(label_vec+N_CLASSES); continue
        init.requires_grad = True
        copyz = 10; optimizer = torch.optim.AdamW([init] , lr=lr); zs = []
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        while True:
            copy = deepcopy(init.data)
            res = model(init)
            z = l(res.view(1,-1), label_vec)
            # if iteration % 1000 == 0: plot_image(init.data.view(1,28,28).numpy().squeeze(), iteration)
            optimizer.zero_grad(); z.backward(); optimizer.step(); scheduler.step()
            init.data = scale(init.data)
            equal = equal + 1 if torch.all(abs(copy - init.data) < 1e-5) or (0 < copyz - z.item() < epsilon) else 0
            if equal > 30: break
            # print(equal, end=' ')
            copyz = z.item()
            # if iteration % 10 == 0: zs.append(copyz)
            iteration += 1
        # plt.plot(zs); plt.show(); plt.clf()
        init.requires_grad = False
        diffs.append(z.item()); data.append(init); labels.append(label_vec+N_CLASSES)
    return data, labels, diffs


