import numpy as np
import os
import linecache as lc

#filelist ='/home/miaoqianwen/AgePre/detTrain'
#filelist ='/home/miaoqianwen/AgePre/detTest'
filelist ='/home/miaoqianwen/AgePre/detVal'

dicimdb = '/home/SSD1/AgeDataBase/imdbAge/imdb_dem'
dicwiki = '/home/SSD1/AgeDataBase/wikiAge/wiki_dem'

arr = []

with open(dicimdb) as lmfile:
    lineNum=sum(1 for _ in lmfile)

it=iter(range(1, lineNum))
for m in it:
    line = lc.getline(dicimdb, m)
    line = line.rstrip('\n')
    file = line.split(' ')
    Gender = file[1]
    file = file[0].split('/')
    ImgName = '/home/SSD1/AgeDataBase/imdbAge/imdbMTCNN/' + file[1]
    arr.append(ImgName)
    arr.append(Gender)

with open(dicwiki) as lmfile:
    lineNum=sum(1 for _ in lmfile)

it=iter(range(1, lineNum))
for m in it:
    line = lc.getline(dicwiki, m)
    line = line.rstrip('\n')
    file = line.split(' ')
    Gender = file[1]
    file = file[0].split('/')
    ImgName = '/home/SSD1/AgeDataBase/wikiAge/wikiMTCNN/' + file[1]
    arr.append(ImgName)
    arr.append(Gender)

with open(filelist) as lmfile:
    lineNum=sum(1 for _ in lmfile)

it=iter(range(1, lineNum))
for m in it:
    line = lc.getline(filelist, m)
    line = line.rstrip('\n')
    file = line.split(' ')
    ImgName = file[0]
    try:
        idx = arr.index(ImgName)
        gender = arr [idx+1]
        iAge = file[1]
        newline = ImgName + " " + iAge + " " + gender
        with open('detValG',"a+") as outfile:
            outfile.write(newline + "\n")
    except:
        print(ImgName)