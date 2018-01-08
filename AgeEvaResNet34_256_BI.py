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
from dataloaderimdbwikiTest import *
from AgePreModelResNet34_256 import *


torch.cuda.set_device(0)
cwd = os.getcwd()
print(cwd)

model = AgePre()
model.cuda()
#checkpoint = torch.load('best_imdbAgePreV2_CrossEntloss.pth.tar', map_location=lambda storage, loc: storage)
#checkpoint = torch.load('imdbwikiAgePreResNet34_256_CrossEntloss.pth.tar', map_location=lambda storage, loc: storage)
#checkpoint = torch.load('best_imdbwikiAgePreResNet34_256_CrossEntloss.pth.tar', map_location=lambda storage, loc: storage)
#checkpoint = torch.load('best_imdbwikiAgePreReResNet34_256_CrossEntloss.pth.tar', map_location=lambda storage, loc: storage)
checkpoint = torch.load('best_imdbwikiAgePreResNet34Det256_CrossEntloss.pth.tar', map_location=lambda storage, loc: storage)
#checkpoint = torch.load('imdbwikiAgePreResNet34Det256_OESM.pth.tar', map_location=lambda storage, loc: storage)

model.load_state_dict(checkpoint['state_dict'])
#checkpoint['state_dict'].keys()

modelOESM = AgePre()
modelOESM.cuda()
checkpoint = torch.load('imdbwikiAgePreResNet34Det256_OESM.pth.tar', map_location=lambda storage, loc: storage)
modelOESM.load_state_dict(checkpoint['state_dict'])

#with open("/home/miaoqianwen/AgePre/DataAgeTest") as lmfile:
#    lineNum=sum(1 for _ in lmfile)

with open("/home/miaoqianwen/AgePre/detTest") as lmfile:
    lineNum=sum(1 for _ in lmfile)

it=iter(range(1, lineNum))
for m in it:
#    m = 1
    # line = lc.getline("/home/miaoqianwen/AgePre/DataAgeTest", m)
    line = lc.getline("/home/miaoqianwen/AgePre/detTest", m)
    line = line.rstrip('\n')
    file = line.split(' ')
    ImgName = file[0]
    iAge = int(file[1])
    input = io.imread(ImgName)
    if input.ndim < 3:
        input = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)
    inp = cv2.resize(input, (256, 256))
    #imgI = torch.from_numpy(inp.transpose((2, 0, 1))).float().div(255.0).unsqueeze_(0)
    imgI = (torch.from_numpy(inp.transpose((2, 0, 1))).float().div(255.0).unsqueeze_(0)-0.5)/0.5
    imgI = imgI.cuda()
    imgI = Variable(imgI)
    model.eval()
    agePre = model(imgI)
    v,i =torch.max(agePre[0], 0)
    i=i.cpu().data.numpy()[0]
    modelOESM.eval()
    agePreOESM = modelOESM(imgI)
    vOESM,iOESM =torch.max(agePreOESM[0], 0)
    iOESM=iOESM.cpu().data.numpy()[0]
    print(ImgName)
    print(iAge, ":", i, " ", iOESM)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(inp)
    plt.show()


