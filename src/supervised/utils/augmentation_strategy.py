'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''
import sys
from albumentations.augmentations.transforms import Equalize
sys.path.append('~/matinaMehdizadeh/Magnification-Prior-Self-Supervised-Method-main/src/')

from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from self_supervised_phase1.utils import config

ft_exp_augmentation = A.Compose([
        A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.4),
        A.GaussianBlur((3, 3), (0.1, 2.0)),
        A.Flip(p=0.3),
        A.Rotate(p=0.3),
        A.Affine(translate_percent = 0.05, p=0.3),
        A.Resize(height=341, width=341, p=1),
        A.RandomCrop(height=252,width=252,p=1)
        ])