import torch
import argparse
import os
import numpy as np
import math
import itertools
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
import torchvision.transforms as transforms

from model import *
from make_dataset import ImageDataset

cuda = torch.cuda.is_available()

hr_shape = (720, 720)

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
    ImageDataset("data/sub_sample/", hr_shape=hr_shape),
    batch_size=1,
    shuffle=False,
    num_workers=2,
)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

inv_normalize = transforms.Normalize(
   mean= [-m/s for m, s in zip(mean, std)],
   std= [1/s for s in std]
)


print('Hawk-Eye is on work......')
for i, images in enumerate(tqdm(dataloader)):
    # Configure model input
    imgs_lr = Variable(images["lr"].type(Tensor))
    imgs_hr = Variable(images["hr"].type(Tensor))
    
    generator.eval()
    # Generate a high resolution image from low resolution input
    gen_hr = generator(imgs_lr)
    
    save_image(inv_normalize(imgs_lr),f"reports/predict/lr_{i}.png")
    save_image(inv_normalize(imgs_hr),f"reports/predict/hr_{i}.png")
    save_image(inv_normalize(gen_hr),f"reports/predict/ghr_{i}.png")