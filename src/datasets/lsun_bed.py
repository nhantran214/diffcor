
import torch
from PIL import Image 
import os, json, glob
import cv2
import pandas as pd

from .base_dataset import BaseDataset
from utils import log_print

class lsun_bed(BaseDataset):
    def __init__(self,root,train_type="train",transform=None,num_classes=2, diffusion_type = 'ldm'):
        super(lsun_bed,self).__init__(root=root, transform=transform, num_classes=num_classes)
        cwd = os.getcwd()
        root_full = os.path.join(cwd, root)
        if train_type == "train":
            real_path = str(train_type) + '/'+'real-dd'+ '/diff'
            fake_path = str(train_type) + '/' + str(diffusion_type) + '/diff'
            # real_path = str(train_type) + '/'+'real-dd'+ '/original'
            # fake_path = str(train_type) + '/' + str(diffusion_type) + '/original'
            # real_path = str(train_type) + '/'+'real'
            # fake_path = str(train_type) + '/' + str(diffusion_type)
            real_path_full = os.path.join(root_full, real_path) + '/*.png'
            fake_path_full = os.path.join(root_full, fake_path) + '/*.png'
            real_imgs = glob.glob(real_path_full)
            fake_imgs = glob.glob(fake_path_full)
            print('real pathhhhhhhhh', real_path_full, fake_path_full)
            log_print("[{}]\t fake imgs count :{}, real imgs count :{}".format(train_type, len(fake_imgs),len(real_imgs)))

            fake_imgs = [[p,1] for p in fake_imgs]
            real_imgs = [[p,0] for p in real_imgs]
            self.imgs = fake_imgs + real_imgs
            # print("len self imgs", self.imgs)
        elif train_type == "test":
            real_path = str(train_type) + '/'+'real-dd'+ '/diff'
            fake_path = str(train_type) + '/' + str(diffusion_type) + '/diff'
            # real_path = str(train_type) + '/'+'real-dd'+ '/original'
            # fake_path = str(train_type) + '/' + str(diffusion_type) + '/original'
            # real_path = str(train_type) + '/'+'real'
            # fake_path = str(train_type) + '/' + str(diffusion_type)
            real_path_full = os.path.join(root_full, real_path) + '/*.png'
            fake_path_full = os.path.join(root_full, fake_path) + '/*.png'
            real_imgs = glob.glob(real_path_full)
            fake_imgs = glob.glob(fake_path_full)

            log_print("[{}]\t fake imgs count :{}, real imgs count :{}".format(train_type, len(fake_imgs),len(real_imgs)))

            fake_imgs = [[p,1] for p in fake_imgs]
            real_imgs = [[p,0] for p in real_imgs]
            self.imgs = fake_imgs + real_imgs
            # print("len self imgs", self.imgs)