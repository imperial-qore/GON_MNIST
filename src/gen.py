import random 
import torch
import numpy as np
from copy import deepcopy
from src.utils import *
from src.constants import *
from tqdm import tqdm

def gen(model, data_type, trainloader):
    lr = 0.05
    iteration = 0; equal = 0; label = 2 #random.randint(0,9)
    # label_one_hot = get_one_hot(label, 10)
    print(label); copyz = 10
    diffs, data = [], []
    l, label_vec = torch.nn.CrossEntropyLoss(), torch.LongTensor([label])
    for restart in tqdm(list(range(100)), ncols=80):
        # init = random.choice(trainloader)[0].view(1,1,28,28)
        init = torch.rand((1,1,28,28), dtype=torch.float)*2 - 1
        init.requires_grad = True
        optimizer = torch.optim.Adam([init] , lr=lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        while True:
            copy = deepcopy(init.data)
            res = model(init)
            # z = torch.sum((res - label_one_hot) ** 2)
            z = l(res.view(1,-1), label_vec)
            # if iteration % 1000 == 0: plot_image(init.data.view(1,28,28).numpy().squeeze(), iteration)
            optimizer.zero_grad()
            z.backward()
            optimizer.step()
            equal = equal + 1 if torch.all(abs(copy - init.data) < 1e-5) or z.item() - copyz < 1e-10 else 0
            if equal > 20: break
            copyz = z.item()
            iteration += 1
        diffs.append(z.item()); data.append(copy.view(1,28,28).numpy().squeeze())
    best = data[diffs.index(max(diffs))]
    plot_image(best, label)


