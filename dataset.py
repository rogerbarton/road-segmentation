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
        augment=False,
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

    # do some noise augmentation, on np arrays
    def _augment_noise(self, image, mask, sigma=0.1, snp_prob=0.02):
        # TODO: contrast and brightness adjust
        # gaussian noise:
        # image = (image + (0.01**0.5) * torch.randn(image.size())).clamp_(0, 1)
        image = (
            (image + np.random.normal(0.0, sigma, image.shape))
            .clip(0.0, 1.0)
            .astype(np.float32)
        )
        # salt and pepper noise:
        rnd = np.random.rand(*image.shape)
        image[rnd < snp_prob / 2] = 0.0
        image[rnd > 1 - snp_prob / 2] = 1.0
        return image, mask

    # need tensor or PIL image
    def _augment_brightness_contrast(self, image, mask):
        brightness = np.random.uniform(0.5, 1.5)
        contrast = np.random.uniform(0.5, 1.5)
        hue = np.random.uniform(-0.2, 0.2)

        image = F.adjust_brightness(image, brightness)
        image = F.adjust_contrast(image, contrast)
        image = F.adjust_hue(image, hue)
        return image, mask

    def __getitem__(self, idx):
        image_np = np.array(Image.open(self.img_list[idx])).astype(np.float32) / 255
        mask_np = np.array(Image.open(self.mask_list[idx])).astype(np.float32) / 255
        if self.resize_to != (image_np.shape[0], image_np.shape[1]):  # resize images
            image_np = cv2.resize(image_np, dsize=self.resize_to)
            mask_np = cv2.resize(mask_np, dsize=self.resize_to)
        image_np = np.moveaxis(image_np, -1, 0)  # pytorch needs CHW instead of HWC
        mask_np = np.expand_dims(mask_np, axis=0)  # needs to be 1HW
        if self.augment:
            image, mask = self._augment_brightness_contrast(
                torch.from_numpy(image_np), torch.from_numpy(mask_np)
            )
            image_np = image.numpy()
            mask_np = mask.numpy()
        if self.transforms:
            image_np = self.transforms(image_np)
        if self.augment:
            image_np, mask_np = self._augment_noise(image_np, mask_np)
        image = np_to_tensor(image_np, self.device)
        mask = np_to_tensor(mask_np, self.device)

        # for visualizing:
        # _, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(image.permute(1, 2, 0))
        # axarr[1].imshow(mask[0])
        # plt.show()

        return image, mask
