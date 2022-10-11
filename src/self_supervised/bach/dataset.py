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

        self.image_type_list = [bc_config.X40]

        
        #key pairing for image examples
        self.image_dict_40x = {}
       

        #key pairing for labels
        self.label_binary_dict = {}
        for patient_dir_name in os.listdir(data_path):
          if patient_dir_name == 'Benign' or patient_dir_name == 'Normal' or patient_dir_name == 'Invasive' or patient_dir_name == 'InSitu':
            patient_uid = patient_dir_name
            print(patient_uid)
            if (patient_dir_name == 'Benign') or (patient_dir_name == 'Normal'):
              binary_label = 'B'
            else:
              binary_label = 'M'


            
            
            #record keeping for 40X images
            path_40x = data_path + patient_dir_name +'/'
            for image_name in os.listdir(path_40x):
                if image_name.split('.')[1] == 'tif':
                  image_seq = image_name.split('.')[0]
                  self.image_dict_40x[patient_uid+'_'+ image_seq] = path_40x + image_name
                  #record keeping for binary label
                  self.label_binary_dict[patient_uid+'_'+ image_seq] = binary_label

        self.list_40X = list(self.image_dict_40x.values())
        self.list_40X_label = list(self.image_dict_40x.keys())
       
        self.dict_list = {}
        self.dict_magnification_list = {}
        self.dict_magnification_list[bc_config.X40] = self.list_40X
        
        
        img_list = self.dict_magnification_list[self.image_type_list[0]]
        for magnification_level in self.image_type_list[1: len(self.image_type_list) - 1]:
            img_list = img_list & self.dict_magnification_list[magnification_level]


        self.dict_list_label = {}
        self.dict_magnification_list_label = {}
        self.dict_magnification_list_label[bc_config.X40] = self.list_40X_label
        
        
        img_list_label = self.dict_magnification_list_label[self.image_type_list[0]]
        for magnification_level in self.image_type_list[1: len(self.image_type_list) - 1]:
            img_list_label = img_list_label & self.dict_magnification_list_label[magnification_level]


        self.image_list = img_list
        self.image_list_label = img_list_label

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        
        # image1 = None
        
        # if index < len(self.input_train):
        image1 = np.array(Image.open(self.image_list[index]))
        image2 = np.array(Image.open(self.image_list[index]))
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
            
        if numberAug < 30:
            image2 = self.augmentation_strategy_2(image2, 1,5, 'Default')

        image2 = self.augmentation_strategy_1(image=image2)

        if self.transform:
            transformed_view1 = self.transform(image1)
            transformed_view2 = self.transform(image2['image'])


            return transformed_view1, bc_config.binary_label_dict[self.label_binary_dict[self.image_list_label[index]]], transformed_view2




def get_BreakHis_trainset_loader(train_path, training_method=None, transform = None,target_transform = None, augmentation_strategy = None, pre_processing = None, image_pair=[], pair_sampling_strategy = None):
    # no addtional preprocessing as of now
    dataset = BreakHis_Dataset_SSL(train_path, transform, augmentation_strategy, pre_processing)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, drop_last=True)
    return train_loader