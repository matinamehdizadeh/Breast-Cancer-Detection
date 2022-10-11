'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

import argparse
import logging
import os, sys


import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append('/content/drive/MyDrive/matinaMehdizadeh/Magnification-Prior-Self-Supervised-Method-main/src/')

from self_supervised.core import models
from self_supervised.pret import trainer_histo
from self_supervised.pret import ssl_loss, pretrain

sys.path.append(os.path.dirname(__file__))
from self_supervised.apply import config, transform, augmentation_strategy
from self_supervised.pret import datasetMag, datasets
from self_supervised.bach import dataset

sys.path.append(os.path.dirname(__file__))

os.environ["KMP_WARNINGS"] = "FALSE"

import bc_config
import argparse


def MPCS_Effnet_b2(data_fold, LR, epoch, description, model_path=None):
    
    fold_root = data_fold # gives the fold no. Example: bc_config.data_path_fold0 +'train_60/'
    LR = float(LR) # 0.00001
    no_epoch = int(epoch) # 150
    description = description # describe fold infomration, pair strategy - for better record keeping
    GPU = torch.device("cuda:0")
    path =  model_path
    
    # Load BreakHis dataset
    train_loader = dataset.get_BreakHis_trainset_loader(
        train_path=fold_root,
        transform = transform.resize_transform,
        augmentation_strategy = augmentation_strategy.pretrain_augmentation,
        pre_processing= [])


        
    # Get network for pretraining with MLP head
    model = models.EfficientNet_MLP(features_dim=2048, v='b2', mlp_dim=2048)
    if path is not None:
      print('load')
      model.load_state_dict(torch.load(path, map_location=GPU))
    model = model.cuda(GPU)

    # Configure optimizer, schedular, loss, other configurations
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'min',
                                                     factor=0.1,
                                                     patience=50,
                                                     min_lr=5e-4)
    criterion = ssl_loss.SimCLR_loss(gpu=GPU, temperature=0.1)
    epochs = no_epoch

    experiment_description = description
    
    trainer = trainer_histo.Trainer_MPCS(
        experiment_description=experiment_description,
        dataloader=train_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        batch_size=config.batch_size,
        gpu = GPU,
        criterion=criterion)
    trainer.train()





if __name__ == '__main__':
    
    print("MPCS self-supervised pretraining...")
    
    # Create parser and parse input
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--data_fold', type=str, required=True, help='The path for fold of dataset to pretrain on'
    )
    parser.add_argument(
        '--LR', type=float, required=False, default=0.00001, help='Learning Rate'
    )
    parser.add_argument(
        '--epoch', type=int, required=False, default=150, help=' No. of epochs'
    )
    parser.add_argument(
        '--description', type=str, required=False, help=' provide experiment description'
    )
    parser.add_argument(
        '--model_path', type=str, required=False, help=' provide experiment description'
    )
    args = parser.parse_args()
    #MPCS ssl pretraining call
    MPCS_Effnet_b2(args.data_fold, args.LR, args.epoch, args.description, args.model_path)