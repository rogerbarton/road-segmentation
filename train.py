# some basic imports

import math
import os
import re
import cv2
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from random import sample
from PIL import Image
import torchvision.transforms as TF
import torchvision.transforms.functional as F
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from dataset import RoadDataset
import datetime


# some constants
PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road

def count_parameters(net):
    """Count number of trainable parameters in `net`."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def preprocessing():
    for img in glob("training/images/*.png"):
        rnd = random.random()
        image = Image.open(img)
        mask = img.replace('images', 'groundtruth')
        mask2 = Image.open(mask)
        img2 = img[:-4]
        if rnd <= 0.5:
            image2 = F.hflip(image)
            mask3 = F.hflip(mask2)
            img2 = img2 + 'hflip' + '.png'
        else:
            image2 = F.vflip(image)
            mask3 = F.hflip(mask2)
            img2 = img2 + 'vflip' + '.png'
            image2.save(img2)
            maskstr = img2.replace('images', 'groundtruth')
            mask3.save(maskstr)
        for i in range(8):
            rnd = random.random()
            if rnd < 0.5:
                angle = random.randint(45*i, 45*(i+1))
                image2 = F.rotate(image, angle)
                mask3 = F.rotate(mask2, angle)
                img2 = img[:-4]
                img2 = img2 + '_rot' + str(angle) + '.png'
                image2.save(img2)
                maskstr = img2.replace('images', 'groundtruth')
                mask3.save(maskstr)

def load_all_from_path(path):
    # loads all HxW .pngs contained in path as a 4D np.array of shape (n_images, H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    return np.stack([np.array(Image.open(f)) for f in sorted(glob(path + '/*.png'))]).astype(np.float32) / 255.

def load_all_from_path_resize(path, resize_to=(400, 400)):
    return np.stack([cv2.resize(np.array(Image.open(f)), dsize=resize_to) for f in sorted(glob(path + '/*.png'))]).astype(np.float32) / 255.

        

def image_to_patches(images, masks=None):
    # takes in a 4D np.array containing images and (optionally) a 4D np.array containing the segmentation masks
    # returns a 4D np.array with an ordered sequence of patches extracted from the image and (optionally) a np.array containing labels
    n_images = images.shape[0]  # number of images
    h, w = images.shape[1:3]  # shape of images
    assert (h % PATCH_SIZE) + (w % PATCH_SIZE) == 0  # make sure images can be patched exactly

    h_patches = h // PATCH_SIZE
    w_patches = w // PATCH_SIZE
    patches = images.reshape((n_images, h_patches, PATCH_SIZE, h_patches, PATCH_SIZE, -1))
    patches = np.moveaxis(patches, 2, 3)
    patches = patches.reshape(-1, PATCH_SIZE, PATCH_SIZE, 3)
    if masks is None:
        return patches

    masks = masks.reshape((n_images, h_patches, PATCH_SIZE, h_patches, PATCH_SIZE, -1))
    masks = np.moveaxis(masks, 2, 3)
    labels = np.mean(masks, (-1, -2, -3)) > CUTOFF  # compute labels
    labels = labels.reshape(-1).astype(np.float32)
    return patches, labels

def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contiguous().pin_memory().to(device=device, non_blocking=True)


def train(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs):
    # training loop
    logdir = './tensorboard/net'
    writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)

    history = {}  # collects metrics at the end of each epoch

    print('Model created with {} trainable parameters'.format(count_parameters(model)))

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        print("Running epoch " + str(epoch))
        # initialize metric list
        metrics = {'loss': [], 'val_loss': []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_'+k] = []
        
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
        # training
        model.train()
        for (x, y) in pbar:
            optimizer.zero_grad()  # zero out gradients
            y_hat = model(x)  # forward pass
            loss = loss_fn(y_hat, y)
            loss.backward()  # backward pass
            optimizer.step()  # optimize weights

            # log partial metrics
            metrics['loss'].append(loss.item())
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

        # validation
        model.eval()
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in eval_dataloader:
                y_hat = model(x) # forward pass

                loss = loss_fn(y_hat, y)
                
                # log partial metrics
                metrics['val_loss'].append(loss.item())
                for k, fn in metric_fns.items():
                    metrics['val_'+k].append(fn(y_hat, y).item())

        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        for k, v in history[epoch].items():
          writer.add_scalar(k, v, epoch)
        print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()]))

    print('Finished Training')


class ImageDataset(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(self, path, device, use_patches=True, resize_to=(400, 400)):
        self.path = path
        self.device = device
        self.use_patches = use_patches
        self.resize_to = resize_to
        self.x, self.y, self.n_samples = None, None, None
        self._load_data()

    def _load_data(self):  # not very scalable, but good enough for now
        self.x = load_all_from_path(os.path.join(self.path, 'images'))
        self.y = load_all_from_path(os.path.join(self.path, 'groundtruth'))
        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x, self.y)
        elif self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)
            self.y = np.stack([cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0)
        self.x = np.moveaxis(self.x, -1, 1)  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)

    def _preprocess(self, x, y):
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing
        return x, y

    def __getitem__(self, item):
        return self._preprocess(np_to_tensor(self.x[item], self.device), np_to_tensor(self.y[[item]], self.device))
    
    def __len__(self):
        return self.n_samples


class Block(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(out_ch),
                                   nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                   nn.ReLU())

    def forward(self, x):
        return self.block(x)


class DecBlock(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    def __init__(self, in_ch, out_ch, mid_chan):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=mid_chan, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(mid_chan),
                                   nn.Conv2d(in_channels=mid_chan, out_channels=out_ch, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(out_ch),
                                   nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                   nn.ReLU())

    def forward(self, x):
        return self.block(x)

        
class UNet(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs_upcon = chs[::-1][:-1]  # number of channels in the decoder
        dec_chs = (1536, 768, 384, 192)
        mid_chs = (1024, 512, 256, 128)
        out_chs = (512, 256, 128, 64) # number of channels in the decoder
        self.enc_blocks = nn.ModuleList([Block(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])])  # encoder blocks
        self.pool = nn.MaxPool2d(2)  # pooling layer (can be reused as it will not be trained)
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(in_ch, out_ch, 2, 2) for in_ch, out_ch in zip(dec_chs_upcon[:-1], dec_chs_upcon[1:])])  # deconvolution
        self.dec_blocks = nn.ModuleList([DecBlock(in_ch, out_ch, mid_ch) for in_ch, out_ch, mid_ch in zip(dec_chs, out_chs, mid_chs)])  # decoder blocks
        self.head = nn.Sequential(nn.Conv2d(dec_chs_upcon[-1], 1, 1), nn.Sigmoid()) # 1x1 convolution for producing the output

        self.enc_blocks2 = nn.ModuleList([Block(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])])  # encoder blocks

        # self.gcn = GCN(384, 384, 0.3, num_stage=0, node_n=384)

    def forward(self, x):
        # encode
        x2 = x.clone()
        enc_features = []
        enc_features2 = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution

        for block in self.enc_blocks2[:-1]:
            x2 = block(x2)  # pass through the block
            enc_features2.append(x2)  # save features for skip connections
            x2 = self.pool(x2)  # decrease resolution
        
        x = self.enc_blocks[-1](x)
        # decode
        for block, upconv, feature, feature2 in zip(self.dec_blocks, self.upconvs, enc_features[::-1], enc_features2[::-1]):
            x = upconv(x)  # increase resolution
            x = torch.cat([x, feature, feature2], dim=1)  # concatenate skip features
            x = block(x)  # pass through the block

        return self.head(x) #.squeeze(dim=1)  # reduce to 1 channel,
        
        # retval = self.gcn(pre_postproc).unsqueeze(dim=1)

        # return retval


def patch_accuracy_fn(y_hat, y):
    # computes accuracy weighted by patches (metric used on Kaggle for evaluation)
    h_patches = y.shape[-2] // PATCH_SIZE
    w_patches = y.shape[-1] // PATCH_SIZE
    patches_hat = y_hat.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    patches = y.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    return (patches == patches_hat).float().mean()

def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()


def main():
    random.seed(17)

    # paths to training and validation datasets
    train_path = 'training'
    val_path = 'validation'
    test_path = 'test'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # reshape the image to simplify the handling of skip connections and maxpooling
    train_dataset = RoadDataset(train_path, device, augment=True, resize_to=(384, 384))
    val_dataset = RoadDataset(val_path, device, augment=False, resize_to=(384, 384))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=True)
    model = UNet().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = 40

    #try:
    train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs)
    #except Exception as e:
    #    print(e)
    #finally:
    print("saving model")
    torch.save(model.state_dict(), ("model_" + str(datetime.datetime.now()) + ".pth"))


if __name__ == '__main__':
    main()
