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
from Randaugment.randaugment import distort_image_with_randaugment

import bc_config

class BreakHis_Dataset_SSL(nn.Module):

    #Default pair sampling - Ordered Pair
    def __init__(self, data_path, transform = None, augmentation_strategy = None, pre_processing = []):
        
        self.transform = transform
        # no preprocessing as of now
        self.pre_processing = pre_processing
        self.augmentation_strategy_1 = augmentation_strategy
        self.augmentation_strategy_2 = distort_image_with_randaugment
        #preprocessing - not in use now
        
        #key pairing for labels
        # self.image_path = []
        # self.path = data_path
        # for im in os.listdir(self.path):
        #     self.image_path.append(self.path+im)

        f = h5py.File('/home/pcam/training_split.h5', 'r')
        self.input_train = f['x']
        # f = h5py.File('/home/pcam/test_split.h5', 'r')
        # self.input_test = f['x']
        # f = h5py.File('/home/pcam/validation_split.h5', 'r')
        # self.input_val = f['x']
        #self.image_path = np.array(self.image_path) 

    def __len__(self):
        return len(self.input_train)

    def __getitem__(self, index):
        
        # image1 = None
        
        # if index < len(self.input_train):
        image1 = self.input_train[index]
        # elif index < len(self.input_train) + len(self.input_test):
        #     indT = index - len(self.input_train)
        #     indT += (14-(indT % 14))
        #     image1 = self.input_test[indT]
        # elif index < len(self.input_train) + len(self.input_test) + len(self.input_val):
        #     indT = index - len(self.input_train) - len(self.input_test)
        #     indT += (14-(indT % 14))
        #     image1 = self.input_val[indT]
        state = torch.get_rng_state()
        numberAug = random.randint(1, 100)          
        image2 = image1.copy()    
        # if numberAug < 40:
        #     image2 = self.augmentation_strategy_2(image1, 1,5, 'Default')

        image2 = self.augmentation_strategy_1(image=image2)

        if self.transform:
            transformed_view1 = self.transform(image1)
            transformed_view2 = self.transform(image2['image'])


            return transformed_view1, transformed_view2




def get_BreakHis_trainset_loader(train_path, transform = None, augmentation_strategy = None, pre_processing = None):
    # no addtional preprocessing as of now
    dataset = BreakHis_Dataset_SSL(train_path, transform, augmentation_strategy, pre_processing)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, drop_last=True)
    return train_loader