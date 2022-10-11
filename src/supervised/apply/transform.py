'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

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
        #t.Resize((224,224)),
        #A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        t.ToTensor()
        ])