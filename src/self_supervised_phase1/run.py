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

sys.path.append(os.path.dirname(__file__))
from self_supervised_phase1.core import models, trainer, sup_con_mod
from self_supervised_phase1.utils import datasets, config, transform, augmentation_strategy
from self_supervised_phase1.bach import dataset
sys.path.append(os.path.dirname(__file__))
os.environ["KMP_WARNINGS"] = "FALSE"

import bc_config
import argparse


def Effnet_b2(data_fold, LR, epoch, description, model_path=None):
    
    fold_root = data_fold # gives the fold no. Example: bc_config.data_path_fold0 +'train_60/'
    LR = float(LR) # 0.00001
    pair_sampling = bc_config.OP
    no_epoch = int(epoch) # 150
    description = description # describe fold infomration, pair strategy - for better record keeping
    GPU = torch.device("cuda:0")
    path = model_path
    
    # Load BreakHis dataset
    
    train_loader = datasets.get_BreakHis_trainset_loader(
        train_path=fold_root,
        training_method='simCLR',
        transform = transform.resize_transform,
        augmentation_strategy = augmentation_strategy.train_augmentation,
        pre_processing= [],
        image_pair=[40,100,200,400],
        pair_sampling_strategy = pair_sampling)
    
        
    # Get network for training with MLP head
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
    criterion = sup_con_mod.SupConMod(gpu=GPU, temperature=0.1)
    epochs = no_epoch

    experiment_description = description
    
    trainer_func = trainer.Trainer(
        experiment_description=experiment_description,
        dataloader=train_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        batch_size=config.batch_size,
        gpu = GPU,
        criterion=criterion)
    trainer_func.train()



if __name__ == '__main__':
    
    print("self-supervised training...")
    
    # Create parser and parse input
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--data_fold', type=str, required=True, help='The path for fold of dataset to train on'
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
    #ssl training call
    Effnet_b2(args.data_fold, args.LR, args.epoch, args.description, args.model_path)
