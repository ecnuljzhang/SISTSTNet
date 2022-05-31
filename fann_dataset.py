#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 23:00:06 2020

@author: lokeshkvn
"""
import datetime
import itertools
import numpy as np
import os
import torchvision.transforms as transforms
from typing import Callable, List, Tuple
from torch.utils.data import Dataset
from PIL import Image
import torch
import glob
import random

TIMESTAMP        = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

RANDOM_SEED      = 99999999

ARCHITECTURE     = 'fannet'

SOURCE_CHARS     = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
TARGET_CHARS     = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

TRAIN_IMAGES_DIR = '/Volumes/Recursion/DL_Resource/STEFANN_CVPR/fannet/train/'
VALID_IMAGES_DIR = '/Volumes/Recursion/DL_Resource/STEFANN_CVPR/fannet/valid/'
PAIRS_IMAGES_DIR = '/Volumes/Recursion/DL_Resource/STEFANN_CVPR/fannet/pairs/'
STANDARD_IMAGES_DIR = '/Volumes/Recursion/DL_Resource/STEFANN_CVPR/fannet/std_font/'
MASKS_DIR = '/Volumes/Recursion/DL_Resource/STEFANN_CVPR/colornet/train/_color_filters/'
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

class GeneratorDataset(Dataset):
    def __init__(
        self,
        source_chars, 
        target_chars, 
        image_dir, 
        std_font_dir,
        masks_dir,
        transform,
        image_ext='.jpg',
        mode='RGB', 
        target_shape=(64, 64), 
        rescale=1.0,    
    ) -> None:
        
        dir_lst = os.listdir(image_dir)
        if '.DS_Store' in dir_lst:
            dir_lst.remove('.DS_Store')
        self._chars = source_chars
        self._perms = list(itertools.product(list(source_chars),
                                             list(target_chars),
                                             dir_lst))
        self._imdir = image_dir
        self._imext = image_ext
        self._std_dir = std_font_dir
        self._imtyp = mode
        self._shape = target_shape
        self._scale = rescale
        self.transforms = transform
        self.color_masks = glob.glob(masks_dir + "/*.jpg")
        self.target_transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    def __len__(self) -> int:
        return len(self._perms)

    def __getitem__(self, idx: int) -> dict:
        ch_src = str(ord(self._perms[idx][0]))
        ch_dst = str(ord(self._perms[idx][1]))
        ch_fnt = self._perms[idx][2]
        im_src = os.path.join(self._imdir, ch_fnt, ch_src + self._imext)
        im_dst = os.path.join(self._imdir, ch_fnt, ch_dst + self._imext)
        im_target = os.path.join(self._std_dir, ch_dst + self._imext)
        mask = random.choice(self.color_masks)
        img_m = Image.open(mask).resize(self._shape)
        img_m = np.asarray(img_m, dtype=np.uint8)
        # print(img_m.shape)
        img_x0 = Image.open(im_src).convert(self._imtyp).resize(self._shape)
        img_x0 = np.asarray(img_x0, dtype=np.uint8)
        img_x0 =  1 * (img_x0 > 64)
        img_x0 = np.repeat(img_x0[...,None],3,axis=2)
        img_x0 = np.asarray(np.multiply(img_x0, img_m), dtype=np.uint8)
        # print(img_x0.shape)
        img_y0 = Image.open(im_dst).convert(self._imtyp).resize(self._shape)
        img_y0 = np.asarray(img_y0, dtype=np.uint8)
        img_y0 = 1 * (img_y0 > 64)
        img_yb = img_y0.copy()
        img_yb = np.atleast_3d(img_yb)
        # print(img_y0.shape)
        img_y0 = np.repeat(img_y0[...,None],3,axis=2)
        img_y0 = np.asarray(np.multiply(img_y0, img_m), dtype=np.uint8)
        # print(img_y0.shape)
        img_tr = Image.open(im_target).convert(self._imtyp).resize(self._shape)
        img_tr = np.asarray(img_tr, dtype=np.uint8)
        # img_tr = 255 * (img_tr > 64)
        img_tr = np.atleast_3d(img_tr)

        if self.transforms:
            img_x0 = self.transforms((img_x0))
            img_y0 = self.transforms((img_y0))
            img_tr = self.target_transform(np.array(img_tr, dtype = np.float32))
            img_yb = self.target_transform(np.array(img_yb, dtype = np.float32))
        
        result = {"source": img_x0}
        result["target"] = img_tr
        result["output"] = img_y0
        result["output_binary"] = img_yb

        return result
    
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
#     ])  

# train_dataset = GeneratorDataset(
#         source_chars=SOURCE_CHARS,
#         target_chars=TARGET_CHARS,
#         image_dir=TRAIN_IMAGES_DIR,
#         std_font_dir = STANDARD_IMAGES_DIR,
#         masks_dir = MASKS_DIR,
#         transform = transform,
#         image_ext=IMAGE_FILE_EXT,
#         mode=IMAGE_READ_MODE,
#         target_shape=INPUT_SHAPE_IMG[:2],
#         rescale=SCALE_COEFF_IMG,
#     ) 