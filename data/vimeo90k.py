   
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import cv2

random.seed(12345)
np.random.seed(12345)


class VimeoTriplet(Dataset):
    def __init__(self, data_root, is_training):
        self.data_root = data_root
        self.image_root = os.path.join(self.data_root, 'sequences')
        self.training = is_training

        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()
        
        self.transforms = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor()
        ])

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
     
        return img0, gt, img1


    def __getitem__(self, index):
        if self.training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
        
        # Load images
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        
        # img1 = Image.open(imgpaths[0])
        # img2 = Image.open(imgpaths[1])
        # img3 = Image.open(imgpaths[2])

        # Data augmentation
        if self.training:
            # seed = random.randint(0, 2**32)
            # random.seed(seed)
            # img1 = self.transforms(img1)
            # random.seed(seed)
            # img2 = self.transforms(img2)
            # random.seed(seed)
            # img3 = self.transforms(img3)
            
            # # Random Temporal Flip
            # if random.random() >= 0.5:
            #     img1, img3 = img3, img1
            #     imgpaths[0], imgpaths[2] = imgpaths[2], imgpaths[0]

            img0, gt, img1 = self.aug(img0, gt, img1, 256, 256)
            # reverse channel
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]

            # vertical flip
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            
            # horizontal flip
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]

            # reverse time
            if random.uniform(0, 1) < 0.5:
                img0, img1 = img1, img0
                imgpaths[0], imgpaths[2] = imgpaths[2], imgpaths[0]

        # else:
            # T = transforms.ToTensor()
            # img1 = T(img1)
            # img2 = T(img2)
            # img3 = T(img3)

    
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1).float() / 255.0 
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1).float() / 255.0
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).float() / 255.0
        
        imgs = [img0, gt, img1]
        
        # imgs = [img1, img2, img3]
        return imgs, imgpaths

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)
        


def get_loader(mode, data_root, batch_size, shuffle, num_workers):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = VimeoTriplet(data_root, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)