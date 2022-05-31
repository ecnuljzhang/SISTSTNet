#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 18:52:56 2020

@author: lokesh
"""

import datetime
from catalyst import utils
import torchvision.transforms as transforms
from utils import LambdaLR
from losses import PixelLoss, StyleLoss, ContentLoss, PSNR, SSIM
import torch

IMAGES_DIR = "/home/lokesh/research/CVPR/TET-E/E"


INPUT_SHAPE_IMG  = (128, 128, 1)


import collections

from tet_dataset import GeneratorTETDataset
from torch.utils.data import DataLoader
from catalyst.contrib.schedulers.onecycle import OneCycleLRWithWarmup

from model import GeneratorModel

from utils import InferMaskCallback, weights_init

def get_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:

    # Creates our train dataset
    train_dataset = GeneratorTETDataset(
        train_path = IMAGES_DIR, dataset_type = "train",
    ) 

    # Creates our valid dataset
    valid_dataset = GeneratorTETDataset(
        train_path = IMAGES_DIR, dataset_type = "train", datasize = 25600
    ) 
    
    # train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement = True, num_samples = 1000000)
    # Catalyst uses normal torch.data.DataLoader
    train_loader = DataLoader(
      train_dataset,
      # sampler = train_sampler,
      batch_size=batch_size,
      num_workers=num_workers,
      drop_last=True,
    )
    
    # valid_sampler = torch.utils.data.RandomSampler(valid_dataset, replacement = True, num_samples = 500000)
    
    valid_loader = DataLoader(
      valid_dataset,
      # sampler = valid_sampler,
      batch_size=batch_size,
      num_workers=num_workers,
      drop_last=True,
    )

    # And excpect to get an OrderedDict of loaders
    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders


batch_size = 32

print(f"batch_size: {batch_size}")

loaders = get_loaders(
    batch_size=batch_size
)

model = GeneratorModel(target_channels = 3, activ_tanh = True)
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
logdir = "./logs/proposed_tet_E_4/"

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
        metrics={"loss_pixel_o": 1.0, "loss_style": 0.5, "loss_content_o": 0.25, "loss_pixel_b": 0.0, "loss_content_b": 0.0},
    ),
    InferMaskCallback(
        out_dir = logdir + "epoch_infer/",
        out_prefix = "intermediate",
        )
    ]

# checkpoint = torch.load("/home/lokesh/research/CVPR/Proposed/logs/proposed_fann__color_10/checkpoints/best_full.pth")['model_state_dict']
# model.load_state_dict(checkpoint)

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    # our dataloaders
    # resume = "/home/lokesh/research/CVPR/Proposed/logs/proposed_fann__color_10/checkpoints/best_full.pth",
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