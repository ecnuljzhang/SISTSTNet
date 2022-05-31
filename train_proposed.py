#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:09:23 2020

@author: lokeshkvn
"""

import datetime
from catalyst import utils
import torchvision.transforms as transforms
from utils import LambdaLR
from losses import PixelLoss, StyleLoss, ContentLoss, PSNR, SSIM, SSIM_Loss
from piq import SSIMLoss
import torch

TRAIN_IMAGES_DIR = '/data/zlj/CSVT/train/'
VALID_IMAGES_DIR = '/data/zlj/CSVT/valid/'
# PAIRS_IMAGES_DIR = '.../LITST-Dataset/pairs/'
STANDARD_IMAGES_DIR = '/data/zlj/CSVT/std_font/'
MASKS_DIR = '/data/zlj/CSVT/color_masks/'


IMAGE_FILE_EXT   = '.jpg'
IMAGE_READ_MODE  = 'L'

INPUT_SHAPE_IMG  = (64, 64, 1)


import collections

from proposed_dataset import GeneratorDataset
from torch.utils.data import DataLoader
from catalyst.contrib.nn.schedulers.onecycle import OneCycleLRWithWarmup

from model import GeneratorModel

from utils import InferMaskCallback, weights_init

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
    ])     

def get_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:

    # Creates our train dataset
    train_dataset = GeneratorDataset(
        image_dir=TRAIN_IMAGES_DIR,
        std_font_dir = STANDARD_IMAGES_DIR,
        masks_dir = MASKS_DIR,
        transform = transform,
        image_ext=IMAGE_FILE_EXT,
        mode=IMAGE_READ_MODE,
        target_shape=INPUT_SHAPE_IMG[:2],
        rescale=1,
    )

    # Creates our valid dataset
    valid_dataset = GeneratorDataset(
        image_dir=VALID_IMAGES_DIR,
        std_font_dir = STANDARD_IMAGES_DIR,
        masks_dir = MASKS_DIR,
        transform = transform,
        image_ext=IMAGE_FILE_EXT,
        mode=IMAGE_READ_MODE,
        target_shape=INPUT_SHAPE_IMG[:2],
        rescale=1,
    )
    
    train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement = True, num_samples = 2500000)
    # Catalyst uses normal torch.data.DataLoader
    train_loader = DataLoader(
      train_dataset,
      sampler = train_sampler,
      batch_size=batch_size,
      num_workers=num_workers,
      drop_last=True,
    )
    
    valid_sampler = torch.utils.data.RandomSampler(valid_dataset, replacement = True, num_samples = 50000)
    
    valid_loader = DataLoader(
      valid_dataset,
      sampler = valid_sampler,
      batch_size=batch_size,
      num_workers=num_workers,
      drop_last=True,
    )

    # And excpect to get an OrderedDict of loaders
    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders


batch_size = 48

print(f"batch_size: {batch_size}")

loaders = get_loaders(
    batch_size=batch_size
)

model = GeneratorModel()
model.init_networks(weights_init)
criterion = {
    "pixel_loss": PixelLoss(),
    "style_loss": StyleLoss(),
    "content_loss": ContentLoss(),
    "PSNR": PSNR(), 
    "SSIM": SSIM(),
    "ssim_loss": SSIM_Loss(),
}

from torch import optim

learning_rate = 0.5e-3

model_params = utils.process_model_params(model)

optimizer = torch.optim.Adam(model_params)

scheduler = OneCycleLRWithWarmup(optimizer, num_steps = 80,lr_range=(1e-3,1e-5), init_lr = learning_rate, warmup_fraction = 0.1)

from catalyst.dl import SupervisedRunner

num_epochs = 100
logdir = "./logs/proposed_our_2/"

device = utils.get_device()
print(f"device: {device}")

fp16_params = dict(opt_level="O1")

print(f"FP16 params: {fp16_params}")
 

from catalyst.dl import CriterionCallback, MetricAggregationCallback

# by default SupervisedRunner uses "features" and "targets",
# in our case we get "image" and "mask" keys in dataset __getitem__
runner = SupervisedRunner(device=device, input_key=["target","source"], input_target_key="output", output_key = "generated")

callbacks = [
    # Each criterion is calculated separately.
    CriterionCallback(
        input_key="output",
        output_key = "generated",
        prefix="loss_pixel",
        criterion_key="pixel_loss"
    ),

    
    CriterionCallback(
        input_key="output",
        output_key = "generated",
        prefix="loss_style",
        criterion_key="style_loss"
    ),
    
    CriterionCallback(
        input_key="output",
        output_key = "generated",
        prefix="loss_content",
        criterion_key="content_loss"
    ),
    
    CriterionCallback(
        input_key="output",
        output_key = "generated",
        prefix="loss_ssim",
        criterion_key="ssim_loss"
    ),
    
    CriterionCallback(
        input_key="output",
        output_key = "generated",
        prefix="ssim_score",
        criterion_key="SSIM"
    ),
    
    CriterionCallback(
        input_key="output",
        output_key = "generated",
        prefix="psnr_score",
        criterion_key="PSNR"
    ),
    
    MetricAggregationCallback(
        prefix="loss",
        mode="weighted_sum", # can be "sum", "weighted_sum" or "mean"
        # because we want weighted sum, we need to add scale for each loss
        metrics={"loss_pixel": 10.0, "loss_style": 5, "loss_content": 5, "loss_ssim": 1},
    ),
    InferMaskCallback(
        out_dir = logdir + "epoch_infer/",
        out_prefix = "intermediate",
        )
    ]

checkpoint = torch.load("/home/zlj/CSVT/style_transfer_v1/logs/proposed_our_2/checkpoints/train.9_full.pth")['model_state_dict']
model.load_state_dict(checkpoint)

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    # our dataloaders
    resume = "/home/zlj/CSVT/style_transfer_v1/logs/proposed_our_2/checkpoints/train.9_full.pth",
    loaders=loaders,
    # path to save logs
    logdir=logdir,
    callbacks=callbacks,
    num_epochs=num_epochs,
    # save our best checkpoint by IoU metric
    main_metric="loss",
    # IoU needs to be maximized.
    minimize_metric = True,
    # for FP16. It uses the variable from the very first cell
    fp16=fp16_params,
    # prints train logs
    verbose=True,
)