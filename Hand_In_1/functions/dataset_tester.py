#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


transform = transforms.Compose([transforms.ToTensor()])

batch_size = 4


# Set up the dataset.
dataset = # YOUR DATASET HERE



# Set up the dataset.
trainloader = torch.utils.data.DataLoader(dataset, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          num_workers=2)

# get some images
dataiter = iter(trainloader)
images, labels = dataiter.next()


for i in range(5): #Run through 5 bathes
    images, labels = dataiter.next()
    for image, label in zip(images,labels): # Run through all samples in a batch
        plt.figure()
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        plt.title(label)
        