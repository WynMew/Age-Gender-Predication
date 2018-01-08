import numpy as np
import os
import pickle
from os.path import exists, join, basename, dirname
from os import makedirs, remove
import re
import cv2
import linecache as lc
from skimage import io

with open("/home/HDD4/Database/imdbAge/imdbagelist") as lmfile:
    lineNum=sum(1 for _ in lmfile)

def add(x):
    with open('imdbagesel',"a+") as outfile:
        outfile.write(x + "\n")

it=iter(range(1, lineNum))
for m in it:
    line = lc.getline("/home/HDD4/Database/imdbAge/imdbagelist", m)
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
    if input.shape[0] == input.shape[1]:
        try:
            if iAge > 0 and iAge < 100:
                line = ImgName + " " + str(iAge)
                add(line)

        except:
            ""