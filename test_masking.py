import matplotlib.pyplot as plt
from math import *
import numpy as np
from pathlib import Path
import os
from PIL import Image

import torch
import torchvision.datasets as datasets

import augmentations as aug

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"



transforms = aug.MaskTransform()
masking = aug.Masking(mask_ratio=0.75)


dataset = datasets.ImageFolder("/data1/jeanne/datasets/dogs_cats/train", transforms)
kwargs = dict(batch_size=64, num_workers=10, pin_memory=True)
loader = torch.utils.data.DataLoader(dataset, **kwargs)

images, labels = next(iter(loader))


patch_size = 16

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])



def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3

    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')

    return



def run_one_image(x):
    #x = torch.tensor(img)
    print(x.shape)
    # make it a batch-like
    #x = x.unsqueeze(dim=0)
    #print(x.shape)

    x_masked, mask, _ = masking(x)

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, patch_size**2 *3)  # (N, H*W, p*p*3)
    mask = unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.show()


    

def unpatchify(x):
    p = patch_size
    h = w = int(sqrt(x.shape[1]))
    assert h * w == x.shape[1]

    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h*p, h*p))
    return imgs




run_one_image(images[0])
