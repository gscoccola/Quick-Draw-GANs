# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:14:22 2019

@author: PC
"""

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


#I create a folder for the ouput
os.makedirs("images", exist_ok=True)

#I Set hiperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=50, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval betwen image samples")
parser.add_argument("--c", type=float, default=0.01, help="clipping parameter ")
parser.add_argument("--nc", type=float, default=5, help="critic iterations per epoch")
parser.add_argument("--datasize", type=float, default=30000, help="number of data images")
parser.add_argument("--p", type=float, default=0.5, help="dropout")

opt = parser.parse_args()
print(opt)




img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

#I define the NN architectures

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 2048),
            nn.Linear(2048, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 1024),
            nn.LeakyReLU(0.2, inplace=True),nn.Dropout(opt.p),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),nn.Dropout(opt.p),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),nn.Dropout(opt.p),
            nn.Linear(256,1)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity



#I innitialize the NN
generator = Generator()
critic = Critic()

#
#I load the dataset (in this case octagon) as a npy and convert it to pytorch tensor
#For other labels, just change the name file
data=np.load('full_numpy_bitmap_octagon.npy',
                      mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')

data2=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])(data)

data3=torch.zeros(opt.datasize,1, 28, 28)

for i in range(0,28):
    for l in range(0,opt.datasize):
        data3[l,:,i,:]=data2[:,l,i*28:(i+1)*28]

dataloader=torch.utils.data.DataLoader(dataset=data3,
                                       batch_size=opt.batch_size, shuffle=True)






# #I use the ADAM optimizer
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(critic.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()        
        
        
        for k in range(opt.nc):
        # ---------------------
        #  Critic training (several iterations per epoch)
        # ---------------------

            optimizer_D.zero_grad()

            #Earth mover Loss function

            d_loss = -torch.mean(critic(real_imgs)) + torch.mean(critic(fake_imgs))
            
            d_loss.backward()
            optimizer_D.step()
            
            for p in critic.parameters():
                p.data.clamp_(-opt.c, opt.c)

      
        
        
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        gen_imgs = generator(z)     

        # Generator Loss
        g_loss = -torch.mean(critic(gen_imgs))

        g_loss.backward()
        optimizer_G.step()

      
        

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(-gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)