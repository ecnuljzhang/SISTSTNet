#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 08:11:17 2020

@author: lokesh
"""
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
from losses import PixelLoss, StyleLoss, ContentLoss, PSNR, SSIM
import torch
import datetime
import itertools
import numpy as np
import os
import torchvision.transforms as transforms
from typing import Callable, List, Tuple
from torch.utils.data import Dataset
from PIL import Image
import random
import glob
import cv2
import matplotlib.pyplot as plt

# TRAIN_IMAGES_DIR = '/home/lokesh/research/CVPR/fannet/train/'
# VALID_IMAGES_DIR = '/home/lokesh/research/CVPR/fannet/valid/'
# PAIRS_IMAGES_DIR = '/home/lokesh/research/CVPR/fannet/pairs/'
# STANDARD_IMAGES_DIR = '/home/lokesh/research/CVPR/fannet/std_font/'
# MASKS_DIR = '/home/lokesh/research/CVPR/colornet/train/_color_filters/'

# IMAGE_FILE_EXT   = '.jpg'
# IMAGE_READ_MODE  = 'L'

INPUT_SHAPE_IMG  = (64, 64, 1)


import collections

from proposed_dataset import GeneratorDataset
from torch.utils.data import DataLoader
from catalyst.contrib.nn.schedulers.onecycle import OneCycleLRWithWarmup

from model import GeneratorModel

from utils import InferMaskCallback, weights_init

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
    ])     

target_transform = transforms.Compose([
        transforms.ToTensor(),
        ])

model = GeneratorModel()
model.init_networks(weights_init)

checkpoint = torch.load("/home/lokesh/research/CVPR/Proposed_rg_16gb/Proposed/logs/proposed_our_2/checkpoints/best.pth")['model_state_dict']
model.load_state_dict(checkpoint)


def predict(im_src, im_dst, im_target, mask):
    img_m = Image.open(mask).resize(INPUT_SHAPE_IMG[:2])
    img_m = np.asarray(img_m, dtype=np.uint8)
    # print(img_m.shape)
    img_x0 = Image.open(im_src).convert('L').resize(INPUT_SHAPE_IMG[:2])
    img_x0 = np.asarray(img_x0, dtype=np.uint8)
    img_x0 =  1 * (img_x0 > 127)
    img_x0 = np.repeat(img_x0[...,None],3,axis=2)
    img_x0 = np.asarray(np.multiply(img_x0, img_m), dtype=np.uint8)
    src = img_x0
    # print(img_x0.shape)
    img_y0 = Image.open(im_dst).convert('L').resize(INPUT_SHAPE_IMG[:2])
    img_y0 = np.asarray(img_y0, dtype=np.uint8)
    img_y0 = 1 * (img_y0 > 127)
    img_yb = img_y0.copy()
    img_yb = np.atleast_3d(img_yb)
    # print(img_y0.shape)
    img_y0 = np.repeat(img_y0[...,None],3,axis=2)
    img_y0 = np.asarray(np.multiply(img_y0, img_m), dtype=np.uint8)
    # print(img_y0.shape)
    img_tr = Image.open(im_target).convert('L').resize(INPUT_SHAPE_IMG[:2])
    img_tr = np.asarray(img_tr, dtype=np.uint8)
    # img_tr = 255 * (img_tr > 64)
    img_tr = np.atleast_3d(img_tr)
    
    if transform:
        img_x0 = transform((img_x0))
        img_y0 = transform((img_y0))
        img_tr = target_transform(np.array(img_tr, dtype = np.float32))
        img_yb = target_transform(np.array(img_yb, dtype = np.float32))
        
    model.eval()
    
    predictions, bin_pred = model(target = img_tr.unsqueeze(dim=0), source = img_x0.unsqueeze(dim=0))
    
    predictions = predictions[0,:,:,:].cpu().data
    
    tmp = (predictions.numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
    
    return tmp, src

# pred = predict(im_src, im_dst, im_target, mask)
# cv2.imwrite("/home/lokesh/research/CVPR/words/0932_stefann.jpg", cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR) )
import shutil
save_path = "/home/lokesh/research/CVPR/words/hind-regular/"
os.makedirs(save_path)
font_style_path = "/home/lokesh/research/CVPR/Proposed_dataset/fannet/train/biryani-regular"
masks = glob.glob("/home/lokesh/research/CVPR/colornet/train/_color_filters" + "/*.jpg")
mask = random.choice(masks)
shutil.copyfile(mask, save_path + "mask_" + os.path.basename(mask))
for im_dst in glob.glob(font_style_path + "/*.jpg"):
    im_src = "/home/lokesh/research/CVPR/fannet/train/hind-regular/65.jpg"
    im_target = "/home/lokesh/research/CVPR/fannet/std_font/" + os.path.basename(im_dst)
    pred = predict(im_src, im_dst, im_target, mask)
    
    cv2.imwrite(save_path + os.path.basename(im_dst), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR) )
    

import shutil
path = "/Volumes/Recursion/DL_Resource/STEFANN_CVPR/fannet/train/kurale-regular/112.jpg"
s, c = path.split('/')[-2:]
save_path = "/home/lokesh/research/CVPR/supplementary_results/Others/results/" + s + c[:-4] + "/"
im_src = "/home/lokesh/research/CVPR/fannet/train/" + s + "/" + c
os.makedirs(save_path)
font_style_path = "/home/lokesh/research/CVPR/Proposed_dataset/fannet/train/" + s
masks = glob.glob("/home/lokesh/research/CVPR/colornet/train/_color_filters" + "/*.jpg")
mask = random.choice(masks)
shutil.copyfile(mask, save_path + "mask_" + os.path.basename(mask))
for im_dst in glob.glob(font_style_path + "/*.jpg"):
    im_target = "/home/lokesh/research/CVPR/fannet/std_font/" + os.path.basename(im_dst)
    pred, img_x0 = predict(im_src, im_dst, im_target, mask)
    cv2.imwrite(save_path + os.path.basename(im_dst), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
cv2.imwrite(save_path + 'source.jpg', cv2.cvtColor(img_x0, cv2.COLOR_RGB2BGR))