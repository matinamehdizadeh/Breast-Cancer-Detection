'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

import numpy as np
import json
import argparse
import time
from tqdm import tqdm
import cv2
import logging
import sys, os
from functools import partial
from pathlib import Path

import optuna
import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from sklearn.metrics import f1_score

import sys
sys.path.append('/content/drive/MyDrive/matinaMehdizadeh/Magnification-Prior-Self-Supervised-Method-main/src/')


from efficientnet_pytorch.utils import MemoryEfficientSwish 
from supervised.core.classification_models import classifier
from supervised.apply.datasets import get_BreakHis_data_loader,get_BreakHis_testdata_loader 
from supervised.apply.transform import train_transform, resize_transform
from supervised.apply.augmentation_strategy import ft_exp_augmentation
from supervised.core.models import EfficientNet_Model
from self_supervised.core.models import EfficientNet_MLP
from supervised.core.train_util import Train_Util
# from supervised.core.train import *
import bc_config
from Randaugment.randaugment import distort_image_with_randaugment

def finetune_Effnet_b2(train_data_fold, test_data_fold, magnification, mpcs_pretrained_model_path, LR, epoch, description, percent):
    
    LR = LR  #0.00002
    experiment_description = description
    epoch = epoch
    model_path = mpcs_pretrained_model_path
    
    weight_decay = 5e-3        
    threshold = 0.5 #0.5
    device = 'cuda:0'
    percent = percent
    
    train_loader = get_BreakHis_data_loader(train_data_fold, transform=train_transform, augmentation_strategy=ft_exp_augmentation, pre_processing=[], image_type_list=[magnification], percent=percent)
    test_loader = get_BreakHis_testdata_loader(test_data_fold, transform = resize_transform, pre_processing=[], image_type_list= [magnification])


    def objective(trial):

        configs = {
            'l1' : 10,
            'l2' : 13,
            'l3' : 11,
            'prob' : 0.5,
         #   "lr" : trial.suggest_float("lr", 1e-5, 1e-4, log=True),
            "lr" : LR,
            'decay' : 5e-3,
            "batch_size": 8,
          #  "treshold": trial.suggest_discrete_uniform("treshold", 0.3, 0.6, 0.1)


        }
        # threshold = configs['treshold']
        # file_object = open('sample.txt', 'a')
        # file_object.write(str(configs['treshold']))
        # file_object.write('\n')
   
        # file_object.close()

        writer = SummaryWriter()
        model = classifier(configs["l1"], configs["l2"], configs["l3"], configs["prob"], model_path, device)
        #model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        for param in model.parameters():
          param.requires_grad = True
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=configs["lr"], weight_decay= configs['decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1 ,patience=10, min_lr= 5e-3)

        train_util = Train_Util(experiment_description = experiment_description, epochs = epoch, model=model, device=device, train_loader=train_loader, val_loader=test_loader, optimizer=optimizer, criterion=criterion, batch_size=configs['batch_size'],scheduler=scheduler, num_classes= len(bc_config.binary_label_list), writer=writer, threshold=threshold)
        # train_util.train_and_evaluate()
        accuracy = train_util.train_and_evaluate()

        return accuracy

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=1)

    trial = study.best_trial
    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

    
    # test_acc = test_accuracy(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))

    #1. Initialize downstream model architecture
    #downstream_task_model = classifier(model_path, device)
    # #2. Initialized and load SSL pretraiend model which include backbone, MLP, and head
    # pretrained_model = EfficientNet_MLP()
    # print('Start - loading MPCS pretrained weights')
    # pretrained_model.load_state_dict(torch.load(model_path, map_location=device))
    # print('Stop - loading MPCS pretrained weights')
    # #3. Use backbone part of pretrained model
    # downstream_task_model.model = pretrained_model.backbone
    # file2 = open("file2.txt", "w")
    # for param in downstream_task_model.parameters():
    #     param.requires_grad = True
    #4. Only for EfficientNet - Replace identity modules of dropout, fc, and swish with Efficient last layer respective components and give num_classes output in fc layer
    # downstream_task_model.model._fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(1408, 1)).requires_grad_(True)
    #downstream_task_model.model._swish = MemoryEfficientSwish()
     
    #downstream_task_model = downstream_task_model.to(device)

    # criterion = nn.BCELoss()
    # # optimizer = torch.optim.Adam(downstream_task_model.parameters(), lr=LR, weight_decay= weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1 ,patience=10, min_lr= 5e-3)


    # writer = SummaryWriter()
    # train_util = Train_Util(experiment_description = experiment_description, epochs = epoch, model=downstream_task_model, device=device, train_loader=train_loader, val_loader=test_loader, optimizer=optimizer, criterion=criterion, batch_size=bc_config.batch_size,scheduler=scheduler, num_classes= len(bc_config.binary_label_list), writer=writer, threshold=threshold)
    # train_util.train_and_evaluate()


if __name__ == "__main__":
    print('BreakHis Breast Cancer Dataset - Finetuning for pretrained network')

    # Create parser and parse input
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_data_fold', type=str, required=True, help='The path for fold of trainset - chose data amount based on evaluation'
    )
    parser.add_argument(
        '--test_data_fold', type=str, required=True, help='The path for fold of testset'
    )
    parser.add_argument(
        '--magnification', type=str, required=True, help='Choose magnification for model fine-tuning, Example - 40x'
    )
    parser.add_argument(
        '--model_path', type=str, required=True, help='The path for MPCS pretrained model'
    )
    parser.add_argument(
        '--LR', type=float, required=False, default=0.00002, help='Learning Rate'
    )
    parser.add_argument(
        '--epoch', type=int, required=False, default=200, help=' No. of epochs'
    )
    parser.add_argument(
        '--percent', type=float, required=False, default=200, help=' percentage'
    )
    parser.add_argument(
        '--description', type=str, required=False, help=' provide experiment description'
    )
    args = parser.parse_args()
    finetune_Effnet_b2(args.train_data_fold, args.test_data_fold, args.magnification, args.model_path, args.LR, args.epoch, args.description, args.percent)
