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
parser.add_argument("--n_epochs", type=int, default=20
                    
                    , help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--datasize", type=float, default=50000, help="number of images of data")
parser.add_argument("--classes", type=int, default=5, help="number of classes")
opt = parser.parse_args()
print(opt)



img_shape = ( 1, 28, 28)

#class discriminator(nn.Module):
#    def __init__(__):
#            super(discriminator,__).__init__()
#            
#    
#            __.model = nn.Sequential(
#                nn.Linear(int(np.prod(img_shape)), 512),
#                nn.Sigmoid(),
#                nn.Linear(512, 256),
#                nn.Sigmoid(),
#                nn.Linear(256, 128),
#                nn.Sigmoid(),
#                nn.Linear(128, 10),        
#                nn.Softmax(1),
#        )
#
#    def forward(self, img):
#        img_flat = img.view(img.size(0), -1)
#        y = self.model(img_flat)
#
#        return y
#X = img.view(img.size(0), -1) 
 
class Cnn (torch.nn.Module):
    
    def __init__(__,_):
        super().__init__()
        __.C1=torch.nn.Conv2d(1,14,5,2,2)
        __.C2=torch.nn.Conv2d(14,28,3,1,1)
        __.MP=torch.nn.MaxPool2d(2,2)
        __.L1=torch.nn.Linear(1372,512)
        __.L2=torch.nn.Linear(512,10)
        
    def forward(__,X):
        h1=__.C1(X).relu()
        h2=__.C2(h1)
        h3=__.MP(h2).relu().view(-1,28*7*7)
        print(h3.shape)
        h4=__.L1(h3).tanh()
        y=__.L2(h4).softmax(1)
        return y

data_transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])




train=datasets.ImageFolder(root='data og/train',
                                           transform=data_transform)

train_dataset_loader = torch.utils.data.DataLoader(train,
                                             batch_size=opt.batch_size, shuffle=True,
                                             num_workers=0)
    
model=Cnn(_)
costf=torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD( model.parameters(), 0.1)


    
#model=discriminator()
#costf=torch.nn.CrossEntropyLoss()
#optim = torch.optim.SGD( model.parameters(), 0.5)

for epoch in range (0,opt.n_epochs):
    E=0
    for i, (data, target) in enumerate(train_dataset_loader):
        images=data
        y=model(images)
        
        optim.zero_grad()
        error=costf(y,target)
        error.backward()
        optim.step()
        E += error.item()
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] "
            % (epoch, opt.n_epochs, i, len(train_dataset_loader
                                           
                                           ), E)
        )
    
crossv=datasets.ImageFolder(root='data og/test',
                                           transform=data_transform)

crossv_dataset_loader = torch.utils.data.DataLoader(crossv,
                                             batch_size=opt.batch_size, shuffle=True,
                                             num_workers=0)


rightcross = 0
totalcross = 0
with torch.no_grad():
    for i, (data, target) in enumerate(crossv_dataset_loader):
        
        images=data
        yp=model(images)
        labels=target
        rightcross+=(yp.argmax(dim=1)==labels).sum().item()
        totalcross+=len(labels)

Accuracycross=rightcross/totalcross
print("precision en el crossvalidation", Accuracycross)    
    

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
print("precision de fish", Accuracy3) 

Accuracy4=right4/total4
print("precision de coffee cup", Accuracy4) 

 
    
    
    
    
    
    
    
    
    
    
    
    
    