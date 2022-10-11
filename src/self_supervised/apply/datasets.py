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
from self_supervised.apply import config
from supervised.apply.cutMix import generate_cutmix_image
from Randaugment.randaugment import distort_image_with_randaugment
from self_supervised.apply.elastic_deformation import elastic_transform
import random
import bc_config
from google.colab.patches import cv2_imshow


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

        for patient_dir_name in os.listdir(train_path):
            patient_uid = patient_dir_name.split('-')[1]
            binary_label = patient_dir_name.split('_')[1]
            multi_label = patient_dir_name.split('_')[2]


            #record keeping for 40X images
            path_40x = train_path + patient_dir_name + '/40X/'
            for image_name in os.listdir(path_40x):
                image_seq = image_name.split('.')[0].split('-')[4]
                self.image_dict_40x[patient_uid + '_' + image_seq] = path_40x + image_name
                # record keeping for binary label
                self.label_binary_dict_40x[patient_uid + '_' + image_seq] = binary_label


            # record keeping for 100X images
            path_100x = train_path + patient_dir_name + '/100X/'
            for image_name in os.listdir(path_100x):
                image_seq = image_name.split('.')[0].split('-')[4]
                self.image_dict_100x[patient_uid + '_' + image_seq] = path_100x + image_name
                self.label_binary_dict_100x[patient_uid + '_' + image_seq] = binary_label


            # record keeping for 200X images
            path_200x = train_path + patient_dir_name + '/200X/'
            for image_name in os.listdir(path_200x):
                image_seq = image_name.split('.')[0].split('-')[4]
                self.image_dict_200x[patient_uid + '_' + image_seq] = path_200x + image_name
                self.label_binary_dict_200x[patient_uid + '_' + image_seq] = binary_label


            # record keeping for 400X images
            path_400x = train_path + patient_dir_name + '/400X/'
            for image_name in os.listdir(path_400x):
                image_seq = image_name.split('.')[0].split('-')[4]
                self.image_dict_400x[patient_uid + '_' + image_seq] = path_400x + image_name
                self.label_binary_dict_400x[patient_uid + '_' + image_seq] = binary_label


        #SSL specific
        self.augmentation_strategy_2 = distort_image_with_randaugment
        self.augmentation_strategy_1 = augmentation_strategy
        self.training_method = config.MPCS
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
                
                        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        
        image1_path, image2_path = None,None

        indexMix = np.random.randint(len(self.image_list), size=1)[0]
        while(self.label_binary_dict_40x[self.image_list[index]] != self.label_binary_dict_40x[self.image_list[indexMix]]):
          indexMix = np.random.randint(len(self.image_list), size=1)[0]
        
        #Ordered Pair
        if 'OP' == self.pair_sampling_strategy:
            randon_mgnification = randrange(4)
            if randon_mgnification == 0:
                image1_path = self.image_dict_40x[self.image_list[index]]
                image2_path = self.image_dict_100x[self.image_list[index]]
                # image3_path = self.image_dict_40x[self.image_list[indexMix]]
                # image4_path = self.image_dict_100x[self.image_list[indexMix]]
                image1_binary_label = self.label_binary_dict_40x[self.image_list[index]]
            elif randon_mgnification == 1:
                image1_path = self.image_dict_100x[self.image_list[index]]
                image2_path = self.image_dict_200x[self.image_list[index]]
                # image3_path = self.image_dict_100x[self.image_list[indexMix]]
                # image4_path = self.image_dict_200x[self.image_list[indexMix]]
                image1_binary_label = self.label_binary_dict_100x[self.image_list[index]]
            elif randon_mgnification == 2:
                image1_path = self.image_dict_200x[self.image_list[index]]
                image2_path = self.image_dict_400x[self.image_list[index]]
                # image3_path = self.image_dict_200x[self.image_list[indexMix]]
                # image4_path = self.image_dict_400x[self.image_list[indexMix]]
                image1_binary_label = self.label_binary_dict_200x[self.image_list[index]]
            elif randon_mgnification == 3:
                image1_path = self.image_dict_200x[self.image_list[index]]
                image2_path = self.image_dict_400x[self.image_list[index]]
                # image3_path = self.image_dict_200x[self.image_list[indexMix]]
                # image4_path = self.image_dict_400x[self.image_list[indexMix]]
                image1_binary_label = self.label_binary_dict_200x[self.image_list[index]]
        
        #Random Pair
        if 'RP' == self.pair_sampling_strategy:
            randon_mgnification_1 = randrange(4)
            randon_mgnification_2 = randrange(4)
            #same magnification - not allowed
            while randon_mgnification_1 == randon_mgnification_2:
                randon_mgnification_2 = randrange(4)
            
            if randon_mgnification_1 == 0:
                image1_path = self.image_dict_40x[self.image_list[index]]
                image1_binary_label = self.label_binary_dict_40x[self.image_list[index]]
            elif randon_mgnification_1 == 1:
                image1_path = self.image_dict_100x[self.image_list[index]]
                image1_binary_label = self.label_binary_dict_100x[self.image_list[index]]
            elif randon_mgnification_1 == 2:
                image1_path = self.image_dict_200x[self.image_list[index]]
                image1_binary_label = self.label_binary_dict_200x[self.image_list[index]]
            elif randon_mgnification_1 == 3:
                image1_path = self.image_dict_400x[self.image_list[index]]
                image1_binary_label = self.label_binary_dict_400x[self.image_list[index]]

            if randon_mgnification_2 == 0:
                image2_path = self.image_dict_40x[self.image_list[index]]
            elif randon_mgnification_2 == 1:
                image2_path = self.image_dict_100x[self.image_list[index]]
            elif randon_mgnification_2 == 2:
                image2_path = self.image_dict_200x[self.image_list[index]]
            elif randon_mgnification_2 == 3:
                image2_path = self.image_dict_400x[self.image_list[index]]

        # Fixed Pair
        if 'FP' == self.pair_sampling_strategy:
            image1_path = self.image_dict_200x[self.image_list[index]]
            image2_path = self.image_dict_400x[self.image_list[index]]
            image1_binary_label = self.label_binary_dict_200x[self.image_list[index]]
        
        image1 = np.array(Image.open(image1_path))
        image2 = np.array(Image.open(image2_path))

        transformed_view1, transformed_view2 = None, None
               
        if self.training_method == config.MPCS:
            state = torch.get_rng_state()

            # numberAug = random.randint(1, 100)     
            # if numberAug < 30:
            #     elasticArg1 = image1.shape[1]
            #     elasticArg2 = image2.shape[1]
            #     image1 = elastic_transform(image1, elasticArg1 * 2, elasticArg1 * 0.08, elasticArg1 * 0.08)
            #     image2 = elastic_transform(image2, elasticArg2 * 2, elasticArg2 * 0.08, elasticArg2 * 0.08)


            numberAug = random.randint(1, 100)          
            
            if numberAug < 30:
                image1 = self.augmentation_strategy_2(image1, 1,5, 'Default')
                image2 = self.augmentation_strategy_2(image2, 1,5, 'Default')
            torch.set_rng_state(state)
            transformed_view2 = self.augmentation_strategy_1(image=image2)
            
            transformed_view1 = self.augmentation_strategy_1(image=image1)
            
            
           
            if self.transform:
                transformed_view1 = self.transform(transformed_view1['image'])
                transformed_view2 = self.transform(transformed_view2['image'])

            return transformed_view1, image1_binary_label, transformed_view2




def get_BreakHis_trainset_loader(train_path, training_method=None, transform = None,target_transform = None, augmentation_strategy = None, pre_processing = None, image_pair=[], pair_sampling_strategy = None):
    # no addtional preprocessing as of now
    dataset = BreakHis_Dataset_SSL(train_path, training_method, transform, target_transform, augmentation_strategy, pre_processing, image_pair, pair_sampling_strategy)
    y_train = np.array([dataset[i][1] for i in range(len(dataset))])
    
    class_sample_count = np.array(
       [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    print(class_sample_count)
    file2 = open("file.txt", "w")
    
    file2.write(str(class_sample_count[0])+'\n')
    file2.write(str(class_sample_count[1]))
    file2.close()
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, drop_last=True)
    return train_loader