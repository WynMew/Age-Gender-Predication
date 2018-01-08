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
from dataloaderimdbwikiAgeG import *
from AgeGPreModelResNet34_256 import *


torch.cuda.set_device(0)
cwd = os.getcwd()
print(cwd)

model = AgeGPre()
model.cuda()
#checkpoint = torch.load('best_imdbAgePreV2_CrossEntloss.pth.tar', map_location=lambda storage, loc: storage)
#checkpoint = torch.load('imdbwikiAgePreResNet34_256_CrossEntloss.pth.tar', map_location=lambda storage, loc: storage)
#checkpoint = torch.load('best_imdbwikiAgePreResNet34_256_CrossEntloss.pth.tar', map_location=lambda storage, loc: storage)
#checkpoint = torch.load('best_imdbwikiAgePreReResNet34_256_CrossEntloss.pth.tar', map_location=lambda storage, loc: storage)
checkpoint = torch.load('imdbwikiAgeGPreResNet34Det256_CrossEntloss.pth.tar', map_location=lambda storage, loc: storage)

model.load_state_dict(checkpoint['state_dict'])
#checkpoint['state_dict'].keys()

#with open("/home/miaoqianwen/AgePre/DataAgeTest") as lmfile:
#    lineNum=sum(1 for _ in lmfile)

with open("/home/miaoqianwen/AgePre/detTest") as lmfile:
    lineNum=sum(1 for _ in lmfile)

it=iter(range(1, lineNum))
counter=0
Gcounter=0
diff=0
for m in it:
    line = lc.getline("/home/miaoqianwen/AgePre/detTestG", m)
    line = line.rstrip('\n')
    file = line.split(' ')
    ImgName = file[0]
    iAge = int(file[1])
    iGen = []
    iGen.append(float(file[2]))
    iGen = np.asarray(iGen)
    input = io.imread(ImgName)
    if input.ndim < 3:
        input = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)
    inp = cv2.resize(input, (256, 256))
    imgI = (torch.from_numpy(inp.transpose((2, 0, 1))).float().div(255.0).unsqueeze_(0)-0.5)/0.5
    imgI = imgI.cuda()
    imgI = Variable(imgI)
    model.eval()
    agePre, genderPre = model(imgI)
    v,i =torch.max(agePre[0], 0)
    i=i.cpu().data.numpy()[0]
    gP = genderPre.cpu().data.numpy()[0]
    if gP <0.5:
        print ("Gender Pre: 0")
        if iGen[0] == 0:
            Gcounter = Gcounter +1
    else:
        print ("Gemder Pre: 1")
        if iGen[0] == 1:
            Gcounter = Gcounter +1

    print(ImgName)
    print("label gender", ": ", iGen[0])
    print(iAge, ":", i)
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    #ax.imshow(inp)
    #plt.show()

    print("--------")
    if abs(i - iAge) <= 5:
        counter = counter + 1

    diff = diff + abs(i - iAge)

print(counter)
print(counter/lineNum)
print(Gcounter)
print(Gcounter/lineNum)
print(diff)
print(diff/lineNum)

