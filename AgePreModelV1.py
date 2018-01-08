from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F


class FeatureExtraction(torch.nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.vgg = nn.Sequential(*list(self.vgg.features.children())[:])
        # freeze parameters
        #for param in self.vgg.parameters():
        #    param.requires_grad = False
        # move to GPU
        self.vgg.cuda()

    def forward(self, image_batch):
        return self.vgg(image_batch)

class Classifier(nn.Module):
    def __init__(self, output_dim=100):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU(True),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, output_dim),
        )
        self.fc.cuda()

    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten
        #print(x)
        x = self.fc(x)
        return x

class AgePre(nn.Module):
    def __init__(self):
        super(AgePre, self).__init__()
        self.FeatureExtraction = FeatureExtraction()
        output_dim = 100
        self.classifier = Classifier(output_dim)

    def forward(self, img):
        # do feature extraction
        feature = self.FeatureExtraction(img)
        Age = self.classifier(feature)
        return Age