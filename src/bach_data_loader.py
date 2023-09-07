import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os, random, shutil, csv, copy
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from collections import Counter
from pathlib import Path


root = root = '/content/ICIAR2018_BACH_Challenge/Photos/'

#Description - Benign
benign_list = ['Benign/', 'Normal/']
#Description - Malignant
malignant_list = ['InSitu/', 'Invasive/']
d = {'Benign' : 'B', 'Normal':'B', 'InSitu':'M', 'Invasive':'M'}
count =0
patient_list = []
abstract_category_list = []
concrete_category_list = []

#Access benign categories patients
for benign_type_dir in benign_list:
    p_dir_path = root + benign_type_dir
    for p_id in os.listdir(p_dir_path):
      if p_id.split('.')[1] == 'tif':
        patient_list.append(p_dir_path + p_id)
        count +=1

#Access malignant categories patients
for malignant_type_dir in malignant_list:
    p_dir_path = root + malignant_type_dir
    for p_id in os.listdir(p_dir_path):
      if p_id.split('.')[1] == 'tif':
        patient_list.append(p_dir_path + p_id)
        count +=1

#Random shuffle the list and extract labels
random.shuffle(patient_list)

print(patient_list)
print('patient count', count)

for patient_path in patient_list:
        sub_class = patient_path.split('/')[-2]
        abstract_category_list.append(d[sub_class])
        concrete_category_list.append(sub_class)
        print(patient_path, sub_class)

k_folds = 5
kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)
stat_dict = {}
stat_dict_test = {}
stat_dict_val = {}
data_splits = kfold.split(patient_list,concrete_category_list)

for fold, (train_ids, test_ids) in enumerate(data_splits):
    with open(root +f'/fold{fold}_stat.csv', 'w') as f:
        fold_path = '/home/a_shared_data/Fold_'+ str(fold) + '_' + str(k_folds)
        Path(fold_path).mkdir(parents=True, exist_ok=True)


        print(f'FOLD {fold}')
        print('--------------------------------')
        temp_abstract_category_list = [abstract_category_list[index] for index in train_ids]
        temp_concrete_category_list = [concrete_category_list[index] for index in train_ids]
        temp_patient_list = [patient_list[index] for index in train_ids]

        # #val - 20%
        temp_patient_list_train, temp_patient_list_val, temp_abstract_category_list_train, temp_abstract_category_list_val = train_test_split(temp_patient_list, temp_concrete_category_list ,stratify= temp_concrete_category_list, test_size=0.25)

        temp_abstract_category_list_test = [abstract_category_list[index] for index in test_ids]
        temp_patient_list_test = [patient_list[index] for index in test_ids]

        #train data move
        fold_path_train = fold_path + '/train/'
        Path(fold_path_train).mkdir(parents=True, exist_ok=True)
        Path(fold_path_train+'Benign/').mkdir(parents=True, exist_ok=True)
        Path(fold_path_train+'Normal/').mkdir(parents=True, exist_ok=True)
        Path(fold_path_train+'Invasive/').mkdir(parents=True, exist_ok=True)
        Path(fold_path_train+'InSitu/').mkdir(parents=True, exist_ok=True)
        for patient in temp_patient_list:

            dest = shutil.copy(patient, fold_path_train + patient.split('/')[-2])

        #val data move
        fold_path_val = fold_path + '/val/'
        Path(fold_path_val).mkdir(parents=True, exist_ok=True)
        Path(fold_path_val+'Benign/').mkdir(parents=True, exist_ok=True)
        Path(fold_path_val+'Normal/').mkdir(parents=True, exist_ok=True)
        Path(fold_path_val+'Invasive/').mkdir(parents=True, exist_ok=True)
        Path(fold_path_val+'InSitu/').mkdir(parents=True, exist_ok=True)
        for patient in temp_patient_list:

            dest = shutil.copy(patient, fold_path_val + patient.split('/')[-2])

        #test data move
        fold_path_test = fold_path + '/test/'
        Path(fold_path_test).mkdir(parents=True, exist_ok=True)
        Path(fold_path_test+'Benign/').mkdir(parents=True, exist_ok=True)
        Path(fold_path_test+'Normal/').mkdir(parents=True, exist_ok=True)
        Path(fold_path_test+'Invasive/').mkdir(parents=True, exist_ok=True)
        Path(fold_path_test+'InSitu/').mkdir(parents=True, exist_ok=True)
        for patient in temp_patient_list_test:

            dest = shutil.copy(patient, fold_path_test+ patient.split('/')[-2])
        break
