'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

import os
import torch
import cv2
import torch.nn as nn
from torch.utils.data import sampler
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from glob import glob
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
import glob
import numpy as np
from skimage import io
import json
import cv2, os, random
import matplotlib.pyplot as plt
import pandas as pd
import PIL
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import ToTensor, Resize
import albumentations as A
from skimage import io
import numpy as np
import sys
sys.path.append('~/matinaMehdizadeh/Magnification-Prior-Self-Supervised-Method-main/src/')

from supervised.utils import config
from Randaugment.randaugment import distort_image_with_randaugment

import bc_config

class BreakHis_Dataset(nn.Module):

    def __init__(self, train_path, transform = None, augmentation_strategy = None, pre_processing = [], image_type_list = []):

        # Standard setting for - dataset path, augmentation, trnformations, preprocessing, etc. 
        self.train_path = train_path
        self.transform = transform
        # No preprocessing as of noe
        self.pre_processing = pre_processing
        self.augmentation_strategy = augmentation_strategy
        self.augmentation_strategy_2 = distort_image_with_randaugment
        #self.augmentation_strategy = distort_image_with_randaugment
        self.image_type_list = image_type_list

        
        #key pairing for image examples
        self.image_dict_40x = {}
       

        #key pairing for labels
        self.label_binary_dict = {}

        for patient_dir_name in os.listdir(train_path):
          if patient_dir_name == 'Benign' or patient_dir_name == 'Normal' or patient_dir_name == 'Invasive' or patient_dir_name == 'InSitu':
            patient_uid = patient_dir_name
            print(patient_uid)
            if (patient_dir_name == 'Benign') or (patient_dir_name == 'Normal'):
              binary_label = 'B'
            else:
              binary_label = 'M'

            #record keeping for 40X images
            path_40x = train_path + patient_dir_name +'/'
            for image_name in os.listdir(path_40x):
                if image_name.split('.')[1] == 'tif':
                  image_seq = image_name.split('.')[0]
                  self.image_dict_40x[patient_uid+'_'+ image_seq] = path_40x + image_name
                  #record keeping for binary label
                  self.label_binary_dict[patient_uid+'_'+ image_seq] = binary_label
            

        
        self.list_40X = list(self.image_dict_40x.keys())
       
        self.dict_list = {}
        self.dict_magnification_list = {}
        self.dict_magnification_list[bc_config.X40] = self.list_40X
        
        
        img_list = self.dict_magnification_list[self.image_type_list[0]]
        for magnification_level in self.image_type_list[1: len(self.image_type_list) - 1]:
            img_list = img_list & self.dict_magnification_list[magnification_level]
        self.image_list = img_list
        
        
                        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        
        item_dict = {}
        magnification = None
        patient_id = None
        
        if bc_config.X40 in self.image_type_list:
            patient_id = self.image_list[index].split('_')[0]
            item_dict[bc_config.X40] = PIL.Image.open(self.image_dict_40x[self.image_list[index]])
            magnification = bc_config.X40
                       
        #Uniform augmentation
        state = torch.get_rng_state()
        if None != self.augmentation_strategy:
            for mg_level in list(item_dict.keys()):
                torch.set_rng_state(state)
                numberAug = random.randint(1, 100)          
            
                if numberAug < 30:
                    item_dict[mg_level] = self.augmentation_strategy_2(np.array(item_dict[mg_level]), 1,5, 'Default')
                item_dict[mg_level] = self.augmentation_strategy(image=np.array(item_dict[mg_level]))

        if None != self.transform:
            for mg_level in list(item_dict.keys()):
                if None == self.augmentation_strategy:
                    if 0 == len(self.pre_processing):
                        item_dict[mg_level] = self.transform(np.array(item_dict[mg_level]))
                    else:
                        item_dict[mg_level] = self.transform(item_dict[mg_level])
                else:
                    item_dict[mg_level] = self.transform(item_dict[mg_level]['image'])
        
        return patient_id, magnification, item_dict, bc_config.binary_label_dict[self.label_binary_dict[self.image_list[index]]], 1

def get_BreakHis_data_loader(dataset_path, transform = None, augmentation_strategy = None, pre_processing = None, image_type_list=[], percent = 1):
    dataset = BreakHis_Dataset(train_path = dataset_path, transform = transform, augmentation_strategy = augmentation_strategy, pre_processing = pre_processing, image_type_list = image_type_list)
    print(len(dataset))
    dataset_final, _ = random_split(dataset, (int(len(dataset)*percent), int(len(dataset)) - int(len(dataset)*percent)))
    print(len(dataset_final))

    
    loader = DataLoader(dataset_final, batch_size=config.batch_size, num_workers=1, shuffle=True)
    return loader

def get_BreakHis_testdata_loader(dataset_path, transform = None, pre_processing = None, image_type_list=[]):
    dataset = BreakHis_Dataset(dataset_path, transform = transform, augmentation_strategy = None, pre_processing=pre_processing, image_type_list = image_type_list)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=1)
    return loader