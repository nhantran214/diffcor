from PIL import Image
import numpy as np

import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import torch
import random


class Select_Transform:
    """random select one from the input transform list"""

    def __init__(self, base_transforms):
        self.base_transforms = base_transforms

    def __call__(self, x):
        return self.base_transforms[random.randint(0,len(self.base_transforms)-1)](x)

class ALBU_AUG:
    def __init__(self, base_transform):
        self.transform = base_transform
    
    def __call__(self, x):
        if isinstance(x, Image.Image):
            x = np.asarray(x)
        return self.transform(image=x)['image']

class Two_Path:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        v1 = self.base_transform(x)
        v2 = self.base_transform(x)
        return [v1, v2]

def get_augs(name="base", norm="imagenet", size=299):
    IMG_SIZE = size
    if norm == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif norm == "0.5": # lsun, celebhq, celebdf
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]

    if name == "None":
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std),
        ])
    elif name == "RandomResizedCrop":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(1/1.3, 1.1), ratio=(0.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std),
        ])
    elif name == "RandomErasing":
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.8,scale=(0.03, 0.30), ratio=(0.4, 2.5),inplace=True),
            transforms.Normalize(mean=mean,std=std),
        ])    
    elif name == "RanSelect":
        return Select_Transform([
            transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std),
            ]),
            transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.RandomErasing(p=1.0,scale=(0.02, 0.20), ratio=(0.5, 2.0)),
                transforms.Normalize(mean=mean,std=std),
            ]),
            transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(1/1.3, 1.0), ratio=(0.9,1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std),
            ])
        ])
    else:
        raise NotImplementedError
