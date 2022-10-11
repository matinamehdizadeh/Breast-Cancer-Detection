'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

import os
import torch
import cv2
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from glob import glob
from torch.utils.data import DataLoader, Dataset
import glob
import numpy as np
from skimage import io
from torch.utils.data import DataLoader, Dataset
import json
import cv2, os
import matplotlib.pyplot as plt
import pandas as pd
import PIL
import h5py
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import ToTensor, Resize
import albumentations as A
from skimage import io
import numpy as np
import random
from random import randrange
import sys

sys.path.append('/content/drive/MyDrive/matinaMehdizadeh/Magnification-Prior-Self-Supervised-Method-main/src/')

from self_supervised.apply import config

import bc_config


class BreakHis_Dataset_SSL(nn.Module):

    # Default pair sampling - Ordered Pair
    def __init__(self, data_path, transform=None, augmentation_strategy=None, pre_processing=[]):

        self.transform = transform
        # no preprocessing as of now
        self.pre_processing = pre_processing
        self.augmentation_strategy_1 = augmentation_strategy
        # preprocessing - not in use now

        # key pairing for labels
        self.image_path_20x = {}
        self.image_path_40x = {}
        self.path = data_path
        for im in os.listdir(self.path + '20/'):
            name = im.split('_')
            if len(name) > 4:
                patient = '_'.join(name[:4])
                self.image_path_20x[patient] = self.path + '20/' + im
                self.image_path_40x[patient] = []
            else:
                name = im.split('.')[0]
                self.image_path_20x[name] = self.path + '20/' + im
                self.image_path_40x[name] = []

        for im in os.listdir(self.path + '40/'):
            name = im.split('_')
            if len(name) > 4:
                patient = '_'.join(name[:4])
                self.image_path_40x[patient].append(self.path + '40/' + im)
            else:
                name = im.split('.')[0][:-1]
                self.image_path_40x[name].append(self.path + '40/' + im)
        self.image_list = list(set(self.image_path_20x))


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        image1 = np.array(PIL.Image.open(self.image_path_20x[self.image_list[index]]))
        numberAug = random.randint(0, len(self.image_path_40x[self.image_list[index]])-1)  

        image2 = PIL.Image.open(self.image_path_40x[self.image_list[index]][numberAug])
        image1 = self.augmentation_strategy_1(image=np.array(image1))
        image2 = self.augmentation_strategy_1(image=np.array(image2))
        if self.transform:
            transformed_view1 = self.transform(image1['image'])
            transformed_view2 = self.transform(image2['image'])

            return transformed_view1, transformed_view2


def get_BreakHis_trainset_loader(train_path, transform=None, augmentation_strategy=None, pre_processing=None):
    # no addtional preprocessing as of now
    dataset = BreakHis_Dataset_SSL(train_path, transform, augmentation_strategy, pre_processing)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, drop_last=True)
    return train_loader