from cv2 import transform
from torchvision import transforms as t
from torchvision.transforms.transforms import RandomCrop, Resize
import sys
sys.path.append('/content/drive/MyDrive/matinaMehdizadeh/Magnification-Prior-Self-Supervised-Method-main/src/')

# Dataset input processing - trainset

resize_transform = t.Compose([
        t.ToPILImage(), 
        t.Resize((341, 341)),
        t.ToTensor()
        ])