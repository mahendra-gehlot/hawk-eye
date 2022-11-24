import torch
import numpy as np
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

resume = True

if resume:
    generator.load_state_dict(torch.load('models/generator.pt'))
    discriminator.load_state_dict(torch.load('models/discriminator.pt'))

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
    batch_size=2,
    shuffle=True,
    num_workers=2,
)

# ----------
#  Training
# ----------
EPOCHs = 300
for epoch in range(0, EPOCHs):
    for itr_id, images in enumerate(dataloader):

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
        print(
            f'Epoch: {epoch}/{EPOCHs} Batch ID: {itr_id}/{len(dataloader)} Loss_D: {loss_D.item():.7f} Loss_G: {loss_G.item():.7f}'
        )

        if (itr_id + 1) == len(dataloader):
            # Saving Model
            torch.save(generator.state_dict(), 'models/generator.pt')
            torch.save(discriminator.state_dict(), 'models/discriminator.pt')

            # Log state of Model
            print(f'The Model is being Saved!')

            # Save image grid with up-sampled inputs and SRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr), -1)
            save_image(img_grid,
                       f"reports/training_results/{itr_id}.png",
                       normalize=False)
