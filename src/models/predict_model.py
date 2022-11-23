import torch
import argparse
import os
import numpy as np
import math
import itertools
import sys
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable

from model import *
from make_dataset import ImageDataset

cuda = torch.cuda.is_available()

hr_shape = (512, 512)

# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(3, *hr_shape))
feature_extractor = FeatureExtractor()

load = True

if load:
    generator.load_state_dict(torch.load('models/generator.pt'))
    discriminator.load_state_dict(torch.load('models/discriminator.pt'))
    
# Set feature extractor to inference mode
feature_extractor.eval()


if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("data/video/data", hr_shape=hr_shape),
    batch_size=1,
    shuffle=False,
    num_workers=2,
)


for i, images in enumerate(dataloader):
    # Configure model input
    imgs_lr = Variable(images["lr"].type(Tensor))
    imgs_hr = Variable(images["hr"].type(Tensor))
    
    generator.eval()
    # Generate a high resolution image from low resolution input
    gen_hr = generator(imgs_lr)
    
    save_image(gen_hr,f"reports/predict/{i}.png")