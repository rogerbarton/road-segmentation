import os
import torch
from torch.utils.data import Dataset
from glob import glob
import numpy as np
import torchvision.transforms.functional as F
import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == "cpu":
        return torch.from_numpy(x).cpu()
    else:
        return (
            torch.from_numpy(x)
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
        image_np = np.array(Image.open(self.img_list[idx])).astype(np.float32) / 255
        mask_np = np.array(Image.open(self.mask_list[idx])).astype(np.float32) / 255
        if self.resize_to != (image_np.shape[0], image_np.shape[1]):  # resize images
            image_np = cv2.resize(image_np, dsize=self.resize_to)
            mask_np = cv2.resize(mask_np, dsize=self.resize_to)
        image_np = np.moveaxis(image_np, -1, 0)  # pytorch needs CHW instead of HWC
        mask_np = np.expand_dims(mask_np, axis=0) # needs to be 1HW
        image = np_to_tensor(image_np, self.device)
        mask = np_to_tensor(mask_np, self.device)
        if self.transforms:
            image = self.transforms(image)
        if self.augment:
            image, mask = self._augment(image, mask)
        
        return image, mask
