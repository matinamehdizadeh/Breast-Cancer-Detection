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
        self.image_dict_100x = {}
        self.image_dict_200x = {}
        self.image_dict_400x = {}

        #key pairing for labels
        self.label_binary_dict = {}
        self.label_multi_dict = {}

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
                self.image_dict_40x[patient_uid+'_'+ image_seq] = path_40x + image_name
                #record keeping for binary label
                self.label_binary_dict[patient_uid+'_'+ image_seq] = binary_label
                #record keeping for multi label
                self.label_multi_dict[patient_uid+'_'+ image_seq] = multi_label
            
            #record keeping for 100X images
            path_100x = patient_dir_name + '/100X/'
            for image_name in os.listdir(path_100x):
                image_seq = image_name.split('.')[0].split('-')[4]
                self.image_dict_100x[patient_uid+'_'+ image_seq] = path_100x + image_name
                self.label_binary_dict[patient_uid+'_'+ image_seq] = binary_label
                #record keeping for multi label
                self.label_multi_dict[patient_uid+'_'+ image_seq] = multi_label

            #record keeping for 200X images
            path_200x = patient_dir_name + '/200X/'
            for image_name in os.listdir(path_200x):
                image_seq = image_name.split('.')[0].split('-')[4]
                self.image_dict_200x[patient_uid+'_'+ image_seq] = path_200x + image_name
                self.label_binary_dict[patient_uid+'_'+ image_seq] = binary_label
                #record keeping for multi label
                self.label_multi_dict[patient_uid+'_'+ image_seq] = multi_label

            #record keeping for 400X images
            path_400x = patient_dir_name + '/400X/'
            for image_name in os.listdir(path_400x):
                image_seq = image_name.split('.')[0].split('-')[4]
                self.image_dict_400x[patient_uid+'_'+ image_seq] = path_400x + image_name
                self.label_binary_dict[patient_uid+'_'+ image_seq] = binary_label
                #record keeping for multi label
                self.label_multi_dict[patient_uid+'_'+ image_seq] = multi_label

        
        self.list_40X = list(self.image_dict_40x.keys())
        self.list_100X = list(self.image_dict_100x.keys())
        self.list_200X = list(self.image_dict_200x.keys())
        self.list_400X = list(self.image_dict_400x.keys())
        self.dict_list = {}
        self.dict_magnification_list = {}
        self.dict_magnification_list[bc_config.X40] = self.list_40X
        self.dict_magnification_list[bc_config.X100] = self.list_100X
        self.dict_magnification_list[bc_config.X200] = self.list_200X
        self.dict_magnification_list[bc_config.X400] = self.list_400X
        
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
        if bc_config.X400 in self.image_type_list:
            patient_id = self.image_list[index].split('_')[0]
            item_dict[bc_config.X400] = PIL.Image.open(self.image_dict_400x[self.image_list[index]])
            magnification = bc_config.X400
        if bc_config.X200 in self.image_type_list:
            patient_id = self.image_list[index].split('_')[0]
            item_dict[bc_config.X200] = PIL.Image.open(self.image_dict_200x[self.image_list[index]])
            magnification = bc_config.X200
        if bc_config.X100 in self.image_type_list:
            patient_id = self.image_list[index].split('_')[0]
            item_dict[bc_config.X100] = PIL.Image.open(self.image_dict_100x[self.image_list[index]])
            magnification = bc_config.X100
        if bc_config.X40 in self.image_type_list:
            patient_id = self.image_list[index].split('_')[0]
            item_dict[bc_config.X40] = PIL.Image.open(self.image_dict_40x[self.image_list[index]])
            magnification = bc_config.X40
                       
        #Uniform augmentation
        state = torch.get_rng_state()
        h_e_matrix = 0
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
                h_e_matrix = H_E_Staining(np.array(item_dict[mg_level]))
                h_e_matrix = np.reshape(h_e_matrix, 6)
                h_e_matrix = np.asarray(h_e_matrix)
            

    
        return patient_id, magnification, item_dict, bc_config.binary_label_dict[self.label_binary_dict[self.image_list[index]]], bc_config.multi_label_dict[self.label_multi_dict[self.image_list[index]]], h_e_matrix


def H_E_Staining(img, Io=240, alpha=1, beta=0.15):
    ''' 
    Normalize staining appearence of H&E stained images
    
    Example use:
        see test.py
        
    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity
        
    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''
             
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    
    HERef = np.array([[0.50139589, 0.09750955], [0.83903808, 0.93542442], [0.21122797, 0.33981325]])
    
    maxCRef = np.array([1.9705, 1.0308])
    
    # define height and width of image
    h, w, c = img.shape
    
    # reshape image
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log((img.astype(np.float)+1)/Io)
    
    # remove transparent pixels
    ODhat = OD[~np.any(OD<beta, axis=1)]
        
    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    #eigvecs *= -1
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3])
    
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    """
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
    H[H>255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
    
    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
    E[E>255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    
    """
    return HE


def get_BreakHis_data_loader(dataset_path, transform = None, augmentation_strategy = None, pre_processing = None, image_type_list=[], percent = 1):
    dataset = BreakHis_Dataset(train_path = dataset_path, transform = transform, augmentation_strategy = augmentation_strategy, pre_processing = pre_processing, image_type_list = image_type_list)
    print(len(dataset))
    dataset_final, _ = random_split(dataset, (int(len(dataset)*percent), int(len(dataset)) - int(len(dataset)*percent)))
    print(len(dataset_final))
    y_train = [dataset_final[i][3] for i in range(len(dataset_final))]
                
    class_sample_count = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    print(class_sample_count)
    file2 = open("file.txt", "w")
    
    file2.write(str(class_sample_count[0])+'\n')
    file2.write(str(class_sample_count[1]))
    file2.close()
    
    loader = DataLoader(dataset_final, batch_size=config.batch_size, num_workers=1, shuffle=True)
    return loader

def get_BreakHis_testdata_loader(dataset_path, transform = None, pre_processing = None, image_type_list=[]):
    dataset = BreakHis_Dataset(dataset_path, transform = transform, augmentation_strategy = None, pre_processing=pre_processing, image_type_list = image_type_list)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1)
    return loader