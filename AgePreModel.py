from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models


class FeatureExtraction(torch.nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        # keep feature extraction network up to pool4 (last layer - 7)
        self.vgg = nn.Sequential(*list(self.vgg.features.children())[:-7])
        # freeze parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        # move to GPU
        self.vgg.cuda()

    def forward(self, image_batch):
        return self.vgg(image_batch)

class FeatureRegression(nn.Module):
    def __init__(self, output_dim=100):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(1024, output_dim)
        self.conv.cuda()
        self.linear.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        #print(x)
        x = self.linear(x)
        return x

class AgePre(nn.Module):
    def __init__(self):
        super(AgePre, self).__init__()
        self.FeatureExtraction = FeatureExtraction()
        output_dim = 100
        self.FeatureRegression = FeatureRegression(output_dim)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, img):
        # do feature extraction
        feature = self.FeatureExtraction(img)
        Age = self.FeatureRegression(feature)
        return Age