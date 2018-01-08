import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from dataloaderimdb import *
from AgePreModel import *
import torch.optim as optim
import torch.nn.functional as F
from os.path import exists, join, basename, dirname
from os import makedirs, remove
import shutil
from torch.optim import lr_scheduler
from sklearn.metrics import mutual_info_score
import math

datasetTrain = MyDataSet(filelist ='/home/HDD4/Database/imdbAge/AgeTrain',
            transform=transforms.Compose([
                ToTensorDict(),
                NormalizeImageDict(['image'])
            ]))
dataLoaderTrain = data.DataLoader(datasetTrain, batch_size=50, shuffle=True, num_workers=1)
datasetTest = MyDataSet(filelist ='/home/HDD4/Database/imdbAge/AgeTest',
            transform=transforms.Compose([
                ToTensorDict(),
                NormalizeImageDict(['image'])
            ]))
dataLoaderTest = data.DataLoader(datasetTest, batch_size=50, shuffle=True, num_workers=1)

def save_checkpoint(state, is_best, file):
    model_dir = dirname(file)
    model_fn = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir!='' and not exists(model_dir):
        makedirs(model_dir)
    torch.save(state, file)
    if is_best:
        shutil.copyfile(file, join(model_dir,'best_' + model_fn))

def train(epoch, model, loss_fn, optimizer, dataloader,log_interval=50):
    model.train()
    train_loss = 0
    for i_batch, sample_batched in enumerate(dataloader):
        optimizer.zero_grad()
        img, age = Variable(sample_batched['image'].cuda()), Variable(sample_batched['age'], requires_grad=False)
        ages = age.squeeze(1)
        agesL = ages.type(torch.LongTensor)
        agesL = torch.max(agesL, 1)[1]
        agePre = model(img)
        loss = loss_fn(agePre, agesL.cuda())
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy()[0]
        if i_batch % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                epoch, i_batch , len(dataloader),
                100. * i_batch / len(dataloader), loss.data[0]))
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    return train_loss

def test(model,loss_fn,dataloader):
    model.eval()
    test_loss = 0
    for i_batch, sample_batched in enumerate(dataloader):
        img, age = Variable(sample_batched['image'].cuda()), Variable(sample_batched['age'].cuda(),requires_grad=False)
        ages = age.squeeze(1)
        agesL = ages.type(torch.LongTensor)
        agesL = torch.max(agesL, 1)[1]
        agePre = model(img)
        loss = loss_fn(agePre, agesL.cuda())
        test_loss += loss.data.cpu().numpy()[0]
    test_loss /= len(dataloader)
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    return test_loss


def adjust_lr(optimizer, epoch, maxepoch, init_lr, power = 0.9):
    lr = init_lr * (1-epoch/maxepoch)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


torch.cuda.set_device(2)
cwd = os.getcwd()
print(cwd)

model = AgePre()
model.cuda()
init_lr = 1e-2
optimizer = optim.SGD(model.FeatureRegression.parameters(), lr= init_lr, momentum=0.5)

loss = nn.CrossEntropyLoss()
best_test_loss = float("inf")
print('Starting training...')
resume = 0
start_epoch = 1
end_epoch = 20
if resume:
    checkpoint = torch.load('/home/miaoqianwen/HDD6/FaceAlignment/MyPoseNet/WebFaceAffineRegression3b1Mynet_MSEloss.pth.tar',
                            map_location=lambda storage, loc: storage)
    #checkpoint = torch.load('/home/miaoqianwen/HDD6/FaceAlignment/CNNGeometricPytorch/trained_models/best_pascal_checkpoint_adam_affine_grid_loss.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    #model.parameters
    start_epoch = checkpoint['epoch']
    best_test_loss = checkpoint['best_test_loss']
    optimizer.load_state_dict(checkpoint['optimizer'])


for epoch in range(start_epoch, end_epoch + 1):
    train_loss = train(epoch, model, loss, optimizer, dataLoaderTrain,  log_interval=10)
    test_loss = test(model, loss, dataLoaderTest)
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    lr_now = adjust_lr(optimizer, epoch, end_epoch + 1, init_lr, power=0.9)
    print(lr_now)
    # remember best loss
    is_best = test_loss < best_test_loss
    best_test_loss = min(test_loss, best_test_loss)
    #checkpoint_name = os.path.join('AngleregRessionAlex_mse_loss.pth.tar')
    checkpoint_name = os.path.join('imdbAgePre_CrossEntloss.pth.tar')
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_test_loss': best_test_loss,
        'optimizer': optimizer.state_dict(),
    }, is_best, checkpoint_name)

print('Done!')