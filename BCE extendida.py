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

#Defino hiperparametros
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval betwen image samples")
parser.add_argument("--p", type=float, default=0.5, help="dropout")
parser.add_argument("--datasize", type=float, default=50000, help="number of images of data")
parser.add_argument("--ndiscr", type=int, default=3, help="number of discr training per epoch")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

#Defino la arquitectura (extendida respecto a la del paper)

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
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 2048),
            nn.LeakyReLU(0.2, inplace=True),nn.Dropout(opt.p),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),nn.Dropout(opt.p),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),nn.Dropout(opt.p),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),nn.Dropout(opt.p),
            nn.Linear(256,1),
                nn.Sigmoid(), 
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

#Funcion de costo BCE
adversarial_loss = torch.nn.BCELoss()

#Inicializo las redes
generator = Generator()
discriminator = Discriminator()


#Loadeo los datos (en este caso circle) de un archivo npy y lo paso a tensor torch
data=np.load('full_numpy_bitmap_circle.npy',
                      mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')

data2=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])(data)

data3=torch.zeros(opt.datasize,1, 28, 28)

for i in range(0,28):
    for l in range(0,opt.datasize):
        data3[l,:,i,:]=data2[:,l,i*28:(i+1)*28]

dataloader=torch.utils.data.DataLoader(dataset=data3,
                                       batch_size=opt.batch_size, shuffle=True)

# Optimizadores (ADAM con mismo parametros que el paper original del 2014)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.FloatTensor

# ----------
#  Entrenamiento
# ----------

for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        # Defino real y generado
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configuro input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Entrenamiento del generador
        # -----------------

        optimizer_G.zero_grad()

        # Sampleo de la dimension latente
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Genero imágenes
        gen_imgs = generator(z)

        # Calculo la perdida
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Entro en el discriminador (varias veces por época)
        # ---------------------
        
        for k in range(0,opt.ndiscr):
            optimizer_D.zero_grad()
        
            # Perdida del discriminador
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
        
            d_loss.backward()
            optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        #Guardo samples generadas
        if batches_done % opt.sample_interval == 0:
            save_image(-gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)