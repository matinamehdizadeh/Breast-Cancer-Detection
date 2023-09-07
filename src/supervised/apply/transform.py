from cv2 import transform
from torchvision import transforms as t
import albumentations as A
from albumentations.pytorch import ToTensorV2

from self_supervised.apply import config

# Dataset input processing - trainset
train_transform = t.Compose([
        t.ToPILImage(), 
        t.ToTensor()
        ])

resize_transform = t.Compose([
        t.ToPILImage(),
        t.Resize((341, 341)),
        t.ToTensor()
        ])

resize_transform2 = t.Compose([
        t.ToPILImage(), 
        t.Resize((11, 11)),
        t.ToTensor()
        ])