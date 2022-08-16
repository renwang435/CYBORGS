import os
import random
import sys

import albumentations as alb
import ipdb
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2
from loguru import logger
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

import utils.distributed as dist
import data.transforms as T


def renew_dataloader(args, mask_dir):
    DatasetClass = DetConMaskDataset
    train_dataset = DatasetClass(args.data_dir, mask_dir, 'train2017', args.scramble_masks)

    if dist.get_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = torch.utils.data.sampler.SequentialSampler(train_dataset)
    
    # create train and val dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    return train_sampler, train_loader


class DetConMaskDataset(Dataset):
    def __init__(self, main_dir, mask_dir, split, 
                    scrambled=False, mask_threshold=128):
        self.main_dir = os.path.join(main_dir, split)
        self.mask_dir = os.path.join(mask_dir)
        self.split = split
        self.scrambled = scrambled
        self.mask_threshold = mask_threshold

        all_imgs = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.main_dir)) for f in fn]
        random.shuffle(all_imgs)

        self.total_imgs = all_imgs

        # Data augmentation pipeline
        # self.crop = alb.RandomResizedCrop(height=224, width=224, scale=(0.08, 1), ratio=(3. / 4, 4. / 3), p=1.0)
        self.transform = alb.Compose([
                alb.RandomResizedCrop(height=224, width=224, scale=(0.08, 1), ratio=(3. / 4, 4. / 3), p=1.0),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
                alb.ToGray(p=0.2),
                alb.GaussianBlur(blur_limit=(23, 23), sigma_limit=(0.1, 2.0), p=1.0),
                alb.Solarize(threshold=128, p=0.0),
                alb.HorizontalFlip(p=0.5),
                alb.Normalize(mean=T.IMAGENET_COLOR_MEAN, std=T.IMAGENET_COLOR_STD, p=1.0),
                ToTensorV2(),
            ])
        self.transform_prime = alb.Compose([
            alb.RandomResizedCrop(height=224, width=224, scale=(0.08, 1), ratio=(3. / 4, 4. / 3), p=1.0),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            alb.ToGray(p=0.2),
            alb.GaussianBlur(blur_limit=(23, 23), sigma_limit=(0.1, 2.0), p=0.1),
            alb.Solarize(threshold=128, p=0.2),
            alb.HorizontalFlip(p=0.5),
            alb.Normalize(mean=T.IMAGENET_COLOR_MEAN, std=T.IMAGENET_COLOR_STD, p=1.0),
            ToTensorV2(),
        ])
    
    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        file_name = os.path.basename(self.total_imgs[idx])
        mask_loc = os.path.join(self.mask_dir, file_name).replace('.jpg','.npy')

        # load image
        image = np.array(Image.open(img_loc).convert('RGB'))

        # load masks
        gt_mask = np.load(mask_loc)
        if gt_mask.sum() == 0:
            new_idx = np.random.randint(0, len(self.total_imgs))
            return self.__getitem__(new_idx)

        # binarize mask
        if self.scrambled:
            # Randomize labels in mask
            gt_mask = np.random.randint(0, self.num_classes, size=gt_mask.shape)

        # Crop and augment
        aug1 = self.transform(image=image, mask=gt_mask)
        crop1, mask1 = aug1["image"], aug1["mask"]

        aug2 = self.transform_prime(image=image, mask=gt_mask)
        crop2, mask2 = aug2["image"], aug2["mask"]
        
         # query and key crops
        view1 = [crop1, mask1]
        view2 = [crop2, mask2]    

        return view1, view2
