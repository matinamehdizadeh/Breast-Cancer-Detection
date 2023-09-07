from abc import ABC
from albumentations.augmentations.crops.transforms import RandomResizedCrop
from torchvision import transforms
import albumentations as A
import torch.nn as nn
from albumentations.pytorch import ToTensorV2



pretrain_augmentation = A.Compose([
        A.RandomResizedCrop(height=460,width=700, p=0.3),
        A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.3),
        A.GaussianBlur((3, 3), (0.1, 2.0), p=0.3),
        A.Flip(p=0.3),
        A.Rotate(p=0.3),
        A.Affine(translate_percent = 0.05, p=0.2)
        ])


pretrain_augmentation_original = A.Compose([
        A.RandomResizedCrop(height=460,width=700, p=0.8),
        A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.4),
        A.GaussianBlur((3, 3), (0.1, 2.0), p=0.3),
        A.Flip(p=0.4),
        A.Rotate(p=0.3),
        A.Affine(translate_percent = 0.05, p=0.2)
        ])