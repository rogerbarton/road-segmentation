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

# import scripts.occlusion


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
        if random.random() < 0.5:
            image = (
                (image + np.random.normal(0.0, sigma, image.shape))
                .clip(0.0, 1.0)
                .astype(np.float32)
            )
        # salt and pepper noise:
        if random.random() < 0.5:
            rnd = np.random.rand(*image.shape)
            image[rnd < snp_prob / 2] = 0.0
            image[rnd > 1 - snp_prob / 2] = 1.0
        return image, mask

    # need tensor or PIL image
    def _augment_brightness_contrast(self, image, mask):
        brightness = np.random.uniform(0.75, 1.25)
        contrast = np.random.uniform(0.75, 1.25)
        hue = np.random.uniform(-0.1, 0.1)

        image = F.adjust_brightness(image, brightness)
        image = F.adjust_contrast(image, contrast)
        image = F.adjust_hue(image, hue)
        return image, mask

    def _crop(self, image, mask):
        # images are square anyway
        max_crop_factor = 0.66
        width_original = image.size()[2]
        crop_width_min = int(width_original * max_crop_factor)
        # print(f"width: {width_original}, crop_min: {crop_width_min}")
        top = random.randint(0, width_original - crop_width_min)
        left = random.randint(0, width_original - crop_width_min)
        # height = random.randint(crop_width_min, min(width_original - top, width_original-left))
        height = random.randint(crop_width_min, width_original - top)
        width = random.randint(crop_width_min, width_original - left)
        # maybe not resize
        image = F.resized_crop(
            image, top=top, left=left, height=height, width=width, size=self.resize_to
        )
        # XXX: maybe change interpolation for mask?
        mask = F.resized_crop(
            mask, top=top, left=left, height=height, width=width, size=self.resize_to
        )
        return image, mask

    def _affine(self, image, mask):
        factor = random.uniform(-10, 10)
        angle = random.uniform(-0, 90)
        translate = (random.randint(-25, 25), random.randint(-25, 25))
        # translate = [0, 0]

        image = F.affine(image, scale=1, angle=angle, translate=translate, shear=factor)
        mask = F.affine(mask, scale=1, angle=angle, translate=translate, shear=factor)
        return image, mask

    def __getitem__(self, idx):
        image_np = np.array(Image.open(self.img_list[idx])).astype(np.float32) / 255
        mask_np = np.array(Image.open(self.mask_list[idx])).astype(np.float32) / 255
        image_np = np.moveaxis(image_np, -1, 0)  # pytorch needs CHW instead of HWC
        mask_np = np.expand_dims(mask_np, axis=0)  # needs to be 1HW
        if self.augment:
            image_t = torch.from_numpy(image_np)
            mask_t = torch.from_numpy(mask_np)
            if random.random() < 0.75:
                image_t, mask_t = self._augment_brightness_contrast(image_t, mask_t)
            if random.random() < 0.75:
                image_t, mask_t = self._affine(image_t, mask_t)
            if random.random() < 0.75:
                image_t, mask_t = self._crop(image_t, mask_t)
            else:
                # crop already resized the image
                image_t = F.resize(image_t, self.resize_to)
                mask_t = F.resize(mask_t, self.resize_to)
            image_np = image_t.numpy()
            mask_np = mask_t.numpy()
            # if random.random() < 0.75:
            image_np, mask_np = self._augment_noise(image_np, mask_np)
        if self.transforms:
            image_np = self.transforms(image_np)
        image = np_to_tensor(image_np, self.device)
        mask = np_to_tensor(mask_np, self.device)
        if tuple(image.size()[1:2]) != self.resize_to:
            image = F.resize(image, self.resize_to)
            mask = F.resize(mask, self.resize_to)
        # uncomment for visualizing:
        # _, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(image.permute(1, 2, 0))
        # axarr[1].imshow(mask[0])
        # plt.show()

        return image, mask
