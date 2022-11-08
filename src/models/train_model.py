import torch
import argparse
import os
import numpy as np
import math
import itertools
import sys
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable

from model import *
from make_dataset import ImageDataset

cuda = torch.cuda.is_available()

hr_shape = (512, 512)

# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(3, *hr_shape))
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(),
                               lr=0.0002,
                               betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(),
                               lr=0.0002,
                               betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("data/training", hr_shape=hr_shape),
    batch_size=6,
    shuffle=True,
    num_workers=4,
)

# ----------
#  Training
# ----------
EPOCHs = 400
for epoch in range(0, EPOCHs):
    for i, images in enumerate(dataloader):

        # Configure model input
        imgs_lr = Variable(images["lr"].type(Tensor))
        imgs_hr = Variable(images["hr"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(
            np.ones((imgs_lr.size(0), *discriminator.output_shape))),
                         requires_grad=False)
        fake = Variable(Tensor(
            np.zeros((imgs_lr.size(0), *discriminator.output_shape))),
                        requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n" %
            (epoch, EPOCHs, i, len(dataloader), loss_D.item(), loss_G.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % 100 == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr), -1)
            save_image(img_grid,
                       "reports/GAN_results/%d.png" % batches_done,
                       normalize=False)

torch.save(generator, 'models/generator.pt')