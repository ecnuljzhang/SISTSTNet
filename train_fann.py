#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 23:32:38 2020

@author: lokeshkvn
"""

import datetime
from catalyst import utils
import torchvision.transforms as transforms
from utils import LambdaLR
from losses import PixelLoss, StyleLoss, ContentLoss, PSNR, SSIM
from catalyst.contrib.nn.schedulers.onecycle import OneCycleLRWithWarmup
import torch
TIMESTAMP        = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

RANDOM_SEED      = 99999999

ARCHITECTURE     = 'fannet'

SOURCE_CHARS     = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
TARGET_CHARS     = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

TRAIN_IMAGES_DIR = '/home/lokesh/research/CVPR/fannet/train/'
VALID_IMAGES_DIR = '/home/lokesh/research/CVPR/fannet/valid/'
PAIRS_IMAGES_DIR = '/home/lokesh/research/CVPR/fannet/pairs/'
STANDARD_IMAGES_DIR = '/home/lokesh/research/CVPR/fannet/std_font/'
MASKS_DIR = '/home/lokesh/research/CVPR/colornet/train/_color_filters/'
OUTPUT_DIR       = 'output/{}/{}/'.format(ARCHITECTURE, TIMESTAMP)

IMAGE_FILE_EXT   = '.jpg'
IMAGE_READ_MODE  = 'L'

FUNCTION_OPTIM   = 'adam'
FUNCTION_LOSS    = 'mae'

INPUT_SHAPE_IMG  = (64, 64, 1)
INPUT_SHAPE_HOT  = (len(SOURCE_CHARS), 1)

SCALE_COEFF_IMG  = 1.
BATCH_SIZE       = 64
NUM_EPOCHS       = 10

VERBOSE_LEVEL    = 2

SAVE_IMAGES      = False
SHOW_IMAGES      = True
MAX_IMAGES       = 20

import collections

from fann_dataset import GeneratorDataset
from torch.utils.data import DataLoader

from model import GeneratorModel

from utils import InferMaskCallback, weights_init

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
    ])     

def get_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:

    # Creates our train dataset
    train_dataset = GeneratorDataset(
        source_chars=SOURCE_CHARS,
        target_chars=TARGET_CHARS,
        image_dir=TRAIN_IMAGES_DIR,
        std_font_dir = STANDARD_IMAGES_DIR,
        masks_dir = MASKS_DIR,
        transform = transform,
        image_ext=IMAGE_FILE_EXT,
        mode=IMAGE_READ_MODE,
        target_shape=INPUT_SHAPE_IMG[:2],
        rescale=SCALE_COEFF_IMG,
    )

    # Creates our valid dataset
    valid_dataset = GeneratorDataset(
        source_chars=SOURCE_CHARS,
        target_chars=TARGET_CHARS,
        image_dir=VALID_IMAGES_DIR,
        std_font_dir = STANDARD_IMAGES_DIR,
        masks_dir = MASKS_DIR,
        transform = transform,
        image_ext=IMAGE_FILE_EXT,
        mode=IMAGE_READ_MODE,
        target_shape=INPUT_SHAPE_IMG[:2],
        rescale=SCALE_COEFF_IMG,
    )

    # Catalyst uses normal torch.data.DataLoader
    train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      drop_last=True,
    )

    valid_loader = DataLoader(
      valid_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      drop_last=True,
    )

    # And excpect to get an OrderedDict of loaders
    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders

batch_size = 128

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
    "SSIM": SSIM()
}

from torch import optim

learning_rate = 0.5e-3

model_params = utils.process_model_params(model)

optimizer = torch.optim.Adam(model_params)

scheduler = OneCycleLRWithWarmup(optimizer, num_steps = 50,lr_range=(1e-3,1e-4), init_lr = learning_rate, warmup_fraction = 0.1)

from catalyst.dl import SupervisedRunner

num_epochs = 60
logdir = "./logs/proposed_fann__color_10/"

device = utils.get_device()
print(f"device: {device}")

fp16_params = dict(opt_level="O1")

print(f"FP16 params: {fp16_params}")
 

from catalyst.dl import CriterionCallback, MetricAggregationCallback
# by default SupervisedRunner uses "features" and "targets",
# in our case we get "image" and "mask" keys in dataset __getitem__
runner = SupervisedRunner(device=device, input_key=["target","source"], input_target_key=["output","output_binary"], output_key = ["generated","generated_binary"])

callbacks = [
    # Each criterion is calculated separately.
    CriterionCallback(
        input_key="output",
        output_key = "generated",
        prefix="loss_pixel_o",
        criterion_key="pixel_loss"
    ),
    
    CriterionCallback(
        input_key="output_binary",
        output_key = "generated_binary",
        prefix="loss_pixel_b",
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
        prefix="loss_content_o",
        criterion_key="content_loss"
    ),
    
    CriterionCallback(
        input_key="output_binary",
        output_key = "generated_binary",
        prefix="loss_content_b",
        criterion_key="content_loss"
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
        metrics={"loss_pixel_o": 1.0, "loss_style": 0.5, "loss_content_o": 0.25, "loss_pixel_b": 1.0, "loss_content_b": 0.25},
    ),
    InferMaskCallback(
        out_dir = logdir + "epoch_infer/",
        out_prefix = "intermediate",
        )
    ]


# checkpoint = torch.load("/home/lokesh/research/CVPR/Proposed/logs/proposed_fann__color_7/checkpoints/last.pth")['model_state_dict']
# model.load_state_dict(checkpoint)

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    resume = "/home/lokesh/research/CVPR/Proposed/logs/proposed_fann__color_7/checkpoints/best_full.pth",
    # our dataloaders
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