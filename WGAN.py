# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 19:12:44 2019

@author: PC
"""
import argparse
import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


import math

from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable



#Variables
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=50, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--datasize", type=float, default=50000, help="number of images of data")
parser.add_argument("--classes", type=int, default=5, help="number of classes")
opt = parser.parse_args()
print(opt)



img_shape = ( 1, 28, 28)


data_transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
 
    

generated=datasets.ImageFolder(root='data generada',
                                           transform=data_transform)

generated_dataset_loader = torch.utils.data.DataLoader(generated,
                                             batch_size=1, shuffle=True,
                                             num_workers=0)


right = 0
total = 0
right0 = 0
total0 = 0
right1 = 0
total1 = 0
right2 = 0
total2 = 0
right3 = 0
total3 = 0
right4 = 0
total4 = 0


with torch.no_grad():
    for i, (data, target) in enumerate(generated_dataset_loader):
        

        images=data
        yp=model(images)
        labels=target
        
        right+=(yp.argmax(dim=1)==labels).sum().item()
        total+=len(labels)
        
        if labels==0:
            right0+=(yp.argmax(dim=1)==labels).sum().item()
            total0+=len(labels)
            
        elif labels==1:
            right1+=(yp.argmax(dim=1)==labels).sum().item()
            total1+=len(labels)
        
        elif labels==2:
            right2+=(yp.argmax(dim=1)==labels).sum().item()
            total2+=len(labels)
        
        elif labels==3:
            right3+=(yp.argmax(dim=1)==labels).sum().item()
            total3+=len(labels)
        
        else:
            right4+=(yp.argmax(dim=1)==labels).sum().item()
            total4+=len(labels)
        
        
Accuracy=right/total
print("precision promedio en el dataset generado", Accuracy) 

Accuracy0=right0/total0
print("precision de circle", Accuracy0) 

Accuracy1=right1/total1
print("precision de octagon", Accuracy1) 

Accuracy2=right2/total2
print("precision de cat", Accuracy2) 

Accuracy3=right3/total3
print("precision de fish", Accuracy0) 

Accuracy4=right4/total4
print("precision de coffee cup", Accuracy0)