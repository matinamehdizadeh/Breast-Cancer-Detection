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
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms import ToTensor, Resize
import albumentations as A
from skimage import io
import numpy as np
from random import randrange
import sys

from torchvision.transforms.autoaugment import RandAugment
sys.path.append('/content/drive/MyDrive/matinaMehdizadeh/Magnification-Prior-Self-Supervised-Method-main/src/')
from similarity.utils import config
from Randaugment.randaugment import distort_image_with_randaugment

from similarity.utils.augmentation_strategy import train_augmentation_original
import random
import bc_config


class BreakHis_Dataset_SSL(nn.Module):

    #Default pair sampling - Ordered Pair
    def __init__(self, train_path, training_method=None, transform = None, target_transform = None, augmentation_strategy = None, pre_processing = [], image_pair = [], pair_sampling_strategy = 'OP'):
        
        self.train_path = train_path
        self.transform = transform
        self.target_transform = target_transform
        # no preprocessing as of now
        self.pre_processing = pre_processing
        self.pair_sampling_strategy = pair_sampling_strategy
        self.image_dict_40x = {}
        self.image_dict_100x = {}
        self.image_dict_200x = {}
        self.image_dict_400x = {}
        #preprocessing - not in use now
        
        #key pairing for labels
        self.label_binary_dict_40x = {}
        self.label_binary_dict_100x = {}
        self.label_binary_dict_200x = {}
        self.label_binary_dict_400x = {}
        f = open(train_path, "r")
      
        for patient_dir_name in f.readlines():
            patient_dir_name = patient_dir_name.strip()
            patient_dir = patient_dir_name.split('/')[-1]
            patient_uid = patient_dir.split('-')[1]
            binary_label = patient_dir.split('_')[1]
            multi_label = patient_dir.split('_')[2]


            #record keeping for 40X images
            path_40x = patient_dir_name + '/40X/'
            for image_name in os.listdir(path_40x):
                image_seq = image_name.split('.')[0].split('-')[4]
                self.image_dict_40x[patient_uid + '_' + image_seq] = path_40x + image_name
                # record keeping for binary label
                self.label_binary_dict_40x[patient_uid + '_' + image_seq] = binary_label


            # record keeping for 100X images
            path_100x = patient_dir_name + '/100X/'
            for image_name in os.listdir(path_100x):
                image_seq = image_name.split('.')[0].split('-')[4]
                self.image_dict_100x[patient_uid + '_' + image_seq] = path_100x + image_name
                self.label_binary_dict_100x[patient_uid + '_' + image_seq] = binary_label


            # record keeping for 200X images
            path_200x = patient_dir_name + '/200X/'
            for image_name in os.listdir(path_200x):
                image_seq = image_name.split('.')[0].split('-')[4]
                self.image_dict_200x[patient_uid + '_' + image_seq] = path_200x + image_name
                self.label_binary_dict_200x[patient_uid + '_' + image_seq] = binary_label


            # record keeping for 400X images
            path_400x = patient_dir_name + '/400X/'
            for image_name in os.listdir(path_400x):
                image_seq = image_name.split('.')[0].split('-')[4]
                self.image_dict_400x[patient_uid + '_' + image_seq] = path_400x + image_name
                self.label_binary_dict_400x[patient_uid + '_' + image_seq] = binary_label


        #SSL specific
        f.close()
        self.augmentation_strategy_2 = distort_image_with_randaugment
        self.augmentation_strategy_1 = augmentation_strategy
        self.augmentation_strategy_3 = train_augmentation_original
        self.image_pair = image_pair
        
        self.list_40X = list(self.image_dict_40x.keys())
        self.list_100X = list(self.image_dict_100x.keys())
        self.list_200X = list(self.image_dict_200x.keys())
        self.list_400X = list(self.image_dict_400x.keys())

        self.binary_list_40X = list(self.label_binary_dict_40x.keys())
        self.binary_list_100X = list(self.label_binary_dict_100x.keys())
        self.binary_list_200X = list(self.label_binary_dict_200x.keys())
        self.binary_list_400X = list(self.label_binary_dict_400x.keys())
        temp = list(set(self.list_40X) & set(self.list_100X) & set(self.list_200X) & set(self.list_400X))
        temp__binary_lable = list(set(self.binary_list_40X) & set(self.binary_list_100X) & set(self.binary_list_200X) & set(self.binary_list_400X))
        self.image_list = temp
        self.iter = -1



    def __len__(self):
        self.iter += 1
        return (len(self.image_list))

    def __getitem__(self, index):
        image1_path, image2_path = None,None
       
        self.iterr = int(self.iter/3)
        indexLarge = index
        entend_index = 3
        mag = ''

        randon_mgnification2 = randrange(2)
        random_index = randrange(len(self.image_list))
        if (entend_index == 0):
            image1_path = self.image_dict_40x[self.image_list[index]]
            mag = '40'
            image1_binary_label = self.label_binary_dict_40x[self.image_list[index]]
        elif (entend_index == 1):
            image1_path = self.image_dict_100x[self.image_list[index]]
            mag = '100'
            image1_binary_label = self.label_binary_dict_100x[self.image_list[index]]
        elif (entend_index == 2):
            image1_path = self.image_dict_200x[self.image_list[index]]
            mag = '200'
            image1_binary_label = self.label_binary_dict_200x[self.image_list[index]]
        elif entend_index == 3:
            mag = '400'
            image1_path = self.image_dict_200x[self.image_list[index]]
            image1_binary_label = self.label_binary_dict_200x[self.image_list[index]]

        image2_path = self.image_dict_400x[self.image_list[self.iterr]]

        image1 = np.array(Image.open(image1_path))
        image2 = np.array(Image.open(image2_path))
        transformed_view1, transformed_view2, transformed_view3 = None, None, None
        
        state = torch.get_rng_state()

        numberAug = random.randint(1, 100)          
        
        if numberAug < 30:
            image1 = self.augmentation_strategy_2(image1, 1,5, 'Default')
            

        numberAug = random.randint(1, 100)          
        
        if numberAug < 30:
            image2 = self.augmentation_strategy_2(image2, 1,5, 'Default')
        

        torch.set_rng_state(state)
        transformed_view2 = self.augmentation_strategy_3(image=image2)['image']
        transformed_view1 = self.augmentation_strategy_1(image=image1)['image']

        if self.transform:
            transformed_view1 = self.transform(transformed_view1)
            transformed_view2 = self.transform(transformed_view2)
        
        return transformed_view1, image1_binary_label, self.label_binary_dict_400x[self.image_list[self.iterr]], transformed_view2, self.image_list[self.iterr], self.image_list[index], mag




def get_BreakHis_trainset_loader(train_path, training_method=None, transform = None,target_transform = None, augmentation_strategy = None, pre_processing = None, image_pair=[], pair_sampling_strategy = None):
    dataset = BreakHis_Dataset_SSL(train_path, training_method, transform, target_transform, augmentation_strategy, pre_processing, image_pair, pair_sampling_strategy)
    y_train = np.array([dataset[i][1] for i in range(len(dataset))])
    
    class_sample_count = np.array(
       [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    print(class_sample_count)
    file2 = open("file.txt", "w")
    
    file2.write(str(class_sample_count[0])+'\n')
    file2.write(str(class_sample_count[1]))
    file2.close()
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, drop_last=True)
    return train_loader