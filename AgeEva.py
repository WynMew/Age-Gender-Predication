import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torch.optim as optim
import torch.nn.functional as F
from os.path import exists, join, basename, dirname
from os import makedirs, remove
import shutil
from torch.optim import lr_scheduler
import re
from dataloaderimdb import *
from AgePreModel import *


torch.cuda.set_device(2)
cwd = os.getcwd()
print(cwd)

model = AgePre()
model.cuda()
#checkpoint = torch.load('imdbAgePre_CrossEntloss.pth.tar', map_location=lambda storage, loc: storage)
checkpoint = torch.load('best_imdbAgePre_CrossEntloss.pth.tar', map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])
#checkpoint['state_dict'].keys()

with open("/home/HDD4/Database/imdbAge/AgeEva") as lmfile:
    lineNum=sum(1 for _ in lmfile)

it=iter(range(1, lineNum))
counter=0
diff=0
for m in it:
#    m = 1
    line = lc.getline("/home/HDD4/Database/imdbAge/AgeEva", m)
    line = line.rstrip('\n')
    file = line.split('.')
    ImgName = "/home/HDD4/Database/imdbAge" + file[1] + ".jpg"
    matchDate = re.search(r'nm*\d+_rm\d+_(\d+)-\d+-\d+_(\d+).jpg', ImgName, re.M | re.I)
    yb = matchDate.group(1)
    yt = matchDate.group(2)
    iAge = int(yt) - int(yb)
    input = io.imread(ImgName)
    if input.ndim < 3:
        input = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)
    inp = cv2.resize(input, (128, 128))
    #imgI = torch.from_numpy(inp.transpose((2, 0, 1))).float().div(255.0).unsqueeze_(0)
    imgI = (torch.from_numpy(inp.transpose((2, 0, 1))).float().div(255.0).unsqueeze_(0)-0.5)/0.5
    imgI = imgI.cuda()
    imgI = Variable(imgI)
    model.eval()
    agePre = model(imgI)
    v,i =torch.max(agePre[0], 0)
    i=i.cpu().data.numpy()[0]
    print(iAge, ":", i)
    print("--------")
    if abs(i - iAge) <= 5:
        counter = counter + 1

    diff = diff + abs(i - iAge)

print(counter)
print(counter/lineNum)
print(diff)
print(diff/lineNum)

