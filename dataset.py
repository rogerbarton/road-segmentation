import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from glob import glob
import numpy as np
import torchvision.transforms.functional as F
import random
import matplotlib.pyplot as plt


def image_to_device(path, device):
    if device == "cpu":
        return read_image(path).cpu()
    else:
        return (
            read_image(path)
            .contiguous()
            .pin_memory()
            .to(device=device, non_blocking=True)
        )


class RoadDataset(Dataset):
    def __init__(
        self,
        img_dir,
        device,
        resize_to=(400, 400),
        transforms=None,
        augment=True,
    ):
        self.img_dir = img_dir
        self.transforms = transforms
        self.device = device
        self.augment = augment
        self.resize_to = resize_to
        self.img_list = [x for x in sorted(glob(img_dir + "/images/*.png"))]
        self.mask_list = [y for y in sorted(glob(img_dir + "/groundtruth/*.png"))]

    def __len__(self):
        return len(self.img_list)

    def _augment(self, image, mask):
        # TODO: maybe noise (salt&pepper, gaussian) and colorshift
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
