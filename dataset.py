import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from glob import glob
import numpy as np
import torchvision.transforms.functional as F
import random


def image_to_device(path, device):
    if device == 'cpu':
        return read_image(path).cpu()
    else:
        return read_image(path).contiguous().pin_memory().to(device=device, non_blocking=True)


class RoadDataset(Dataset):
    def __init__(self, img_dir, mask_dir, device, use_patches=True, resize_to=(400, 400), transforms=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.device = device
        self.use_patches = use_patches
        self.resize_to = resize_to
        self.img_list = [x for x in sorted(glob(img_dir + "/images/*.png"))]
        self.mask_list = [y for y in sorted(glob(img_dir + "/groundtruth/*.png"))]

    def __len__(self):
        return len(self.img_list)

    def _augment(self, image, mask):
        # todo, maybe noise and colorshift
        return image, mask

    def __getitem__(self, idx):
        image = image_to_device(self.img_list[idx], self.device)
        mask = image_to_device(self.mask_list[idx], self.device)
        if self.transforms:
            image = self.transforms(image)
        if self.augment:
            image, mask = self._augment(image, mask)
        if self.resize_to:
            image = F.resize(image, self.resize_to)
            mask = F.resize(mask, self.resize_to)
        return image, mask
