import os
import random
import numpy as np
import torch
import PIL
from PIL import Image
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as T

class ImageDataset(Dataset):
    def __init__(self, obj, cfg):
        self.obj = obj
        if obj == 'train':
            self.image_dir = cfg.path.train_image_dir
            self._set_path()
            self.transform = T.Compose([T.ToTensor(),
                                        T.Resize(size=random.randint(128, 156)),
                                        T.RandomCrop(size=(128,128)),
                                        T.Normalize([0.79093, 0.76271, 0.75340], [0.30379, 0.32279, 0.32800])
                                        ])
        elif obj == 'val':
            self.image_dir = cfg.path.val_image_dir
            self._set_path()
            self.transform = T.Compose([T.ToTensor(),
                                        T.Resize(size=128),
                                        T.CenterCrop(size=(128,128)),
                                        T.Normalize([0.79093, 0.76271, 0.75340], [0.30379, 0.32279, 0.32800])
                                        ])
    def _set_path(self):
        self.image_path = []
        if self.obj == 'train':
            for catg in os.listdir(self.image_dir):
                if catg == '.DS_Store':
                    pass
                else:
                    for img in os.listdir(os.path.join(self.image_dir, catg)):
                        self.image_path.append(os.path.join(self.image_dir, catg, img))
        else:
            for img in os.listdir(self.image_dir):
                self.image_path.append(os.path.join(self.image_dir, img))
        
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        img = cv2.imread(self.image_path[index])
        img = self.transform(img)
        img = img
        return img