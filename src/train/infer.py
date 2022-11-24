import torch
import numpy as np
import pandas as pd
import argparse
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import torchvision.transforms as transforms
from model import *
from custom_dataset import ImageDataset

# Adding
parser = argparse.ArgumentParser()
parser.add_argument("input_directory", help="Directory for inference")
parser.add_argument("input_resolution", help="Input Resolution", type=int)
parser.add_argument("output_directory", help="Directory for output")
args = parser.parse_args()

cuda = torch.cuda.is_available()

hr_shape = (args.input_resolution, args.input_resolution)

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
    ImageDataset(args.input_directory, hr_shape=hr_shape),
    batch_size=1,
    shuffle=False,
    num_workers=2,
)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

inv_normalize = transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)],
                                     std=[1 / s for s in std])

image_enlarge = transforms.Resize(hr_shape, Image.BICUBIC)

metrics = []

print('Hawk-Eye is on work......')
for i, images in enumerate(tqdm(dataloader)):
    # Configure model input
    imgs_lr = Variable(images["lr"].type(Tensor))
    imgs_hr = Variable(images["hr"].type(Tensor))

    generator.eval()
    # Generate a high resolution image from low resolution input
    gen_hr = generator(imgs_lr)
    extrapolated_image = image_enlarge(inv_normalize(imgs_lr))
    actual_hr_img = inv_normalize(imgs_hr)
    gen_hr_img = inv_normalize(gen_hr)

    # saving results
    extrapolated_image = make_grid(extrapolated_image, nrow=1, normalize=True)
    gen_hr_img = make_grid(gen_hr_img, nrow=1, normalize=True)
    actual_hr_img = make_grid(actual_hr_img, nrow=1, normalize=True)

    img_grid = torch.cat((extrapolated_image, gen_hr_img, actual_hr_img), -1)

    save_image(img_grid, f"{args.output_directory}inferred_{i}.png")

    # calculating PSNR and SSIM
    psnr_srgan = peak_signal_noise_ratio(actual_hr_img.cpu().detach().numpy(),
                                         gen_hr_img.cpu().detach().numpy())
    # ssim_srgan = structural_similarity(
    #     actual_hr_img.cpu().detach().numpy().squeeze(),
    #     gen_hr_img.cpu().detach().numpy().squeeze(),
    #     channel_axis=3,
    #     multichannel=True)

    psnr_bi = peak_signal_noise_ratio(
        actual_hr_img.cpu().detach().numpy(),
        extrapolated_image.cpu().detach().numpy())

    # ssim_bi = structural_similarity(
    #     actual_hr_img.cpu().detach().numpy().squeeze(),
    #     extrapolated_image.cpu().detach().numpy().squeeze(),
    #     channel_axis=3,
    #     multichannel=True)

    # collecting scores
    scores = [i, psnr_srgan, psnr_bi, 0, 0]
    metrics.append(scores)

# writting results
scores_df = pd.DataFrame(data=metrics,
                         columns=[
                             'IMG_ID', 'PSNR_SRGAN', 'PSNR_BICUBIC',
                             'SSIM_SRGAN', 'SSIM_BICUBIC'
                         ])

scores_df.to_csv('reports/psnr_ssim.csv')
