# some basic imports

import os
import re
import random
import numpy as np
from PIL import Image
import torchvision.transforms as TF
import torchvision.transforms.functional as F
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from dataset import RoadDataset
import datetime
import traceback
import argparse

from models.u_net import UNet as newnet
from models.baselines import ImageDataset, UNet, PatchCNN

# some constants
PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road

def count_parameters(net):
    """Count number of trainable parameters in `net`."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
        

def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contiguous().pin_memory().to(device=device, non_blocking=True)


def train(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs, name):
    # training loop
    logdir = './tensorboard/net'
    writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)

    history = {}  # collects metrics at the end of each epoch

    print('Model created with {} trainable parameters'.format(count_parameters(model)))
    best_val_acc = None
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        print(f"training epoch {epoch}")
        # initialize metric list
        metrics = {'loss': [], 'val_loss': []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_'+k] = []
        
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}', total=len(train_dataloader))
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
        torch.save(model.state_dict(), ("model_temp.pth"))
        print(f"saved epoch {epoch} as model_temp.pht")
        if best_val_acc is None or history[epoch]["val_acc"] > best_val_acc:
            torch.save(model.state_dict(), (name))
            print(f"saved best model so far in epoch {epoch} with val_acc {history[epoch]['val_acc']}")
            best_val_acc = history[epoch]["val_acc"]

    print('Finished Training')


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
    # This is the main training part
    # Here we fix the seeds, call the training loop and Save the model


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="which model to train, choose from [cnn, unet, new_unet]",
        default="new_unet",
    )
    parser.add_argument(
        "--kernel_size",
        help="The kernel size to use, choose from [3, 5, 7]",
        default=3, type=int
    )
    parser.add_argument(
        "--pre_processing",
        help="What preprocessing technique to use, choose from [otf, none, altered_images]",
        default="none"
    )
    parser.add_argument(
        "--loss",
        help="What loss to use, choose from [BCE, BCE_with_logit]",
        default="BCE"
    )
    parser.add_argument(
        "--lr",
        help="What learningrate to use",
        default=0.001, type=float
    )
    parser.add_argument(
        "--optimizer",
        help="What optimizer to use, choose from [adam, SGD, adamax]",
        default="adam"
    )
    parser.add_argument(
        "--output",
        help="Name of the output file",
        default=""
    )
    parser.add_argument("--n_epochs", help="How many epochs to perform", default=40, type=int)
    parser.add_argument("--seed", help="fixed random seed", default=17, type=int)
    parser.add_argument("--bs", help="batch size", default=4, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    # since random is fixed, this gives us some non-random seeds
    np.random.seed(seed=random.randint(0, 100000))
    torch.manual_seed(random.randint(0, 100000))

    # paths to training and validation datasets

    if (args.pre_processing == "none"):
        train_path = 'training'
        val_path = 'validation'
    elif (args.pre_processing == "altered_images" or args.pre_processing == "otf"):
        train_path = 'training_augmented'
        val_path = 'validation_augmented'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # reshape the image to simplify the handling of skip connections and maxpooling
    if (args.pre_processing == "otf"):
        train_dataset = RoadDataset(train_path, device, augment=True, resize_to=(384, 384))
        val_dataset = RoadDataset(val_path, device, augment=False, resize_to=(384, 384))
    else:
        train_dataset = RoadDataset(train_path, device, augment=False, resize_to=(384, 384))
        val_dataset = RoadDataset(val_path, device, augment=False, resize_to=(384, 384))

    if (args.model == "cnn"):
        train_dataset = ImageDataset('training', device)
        val_dataset = ImageDataset('validation', device)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=True)
    elif (args.pre_processing == "none" or args.pre_processing == "altered_images" or args.pre_processing == "otf"):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=True)

    if (args.model == "cnn"):
        model = PatchCNN().to(device)
        metric_fns = {'acc': accuracy_fn}
    elif (args.model == "unet"):
        model = UNet().to(device)
        metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}
    elif (args.model == "new_unet"):
        model = newnet(kernel_size=args.kernel_size).to(device)
        metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}

    if (args.loss == "BCE"):
        loss_fn = nn.BCELoss()
    elif (args.loss == "BCE_with_logit"):
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        print("Plese chose another loss function")
        return -1

    if (args.optimizer == "adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif (args.optimizer == "SGD"):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif (args.optimizer == "adamax"):
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)

    n_epochs = args.n_epochs

    if (args.output == ""):
        model_name = args.model + "_" + str(args.n_epochs) + "_lr=" + str(args.lr) + "_" + args.loss + "_" + args.pre_processing + "_ks=" + str(args.kernel_size)
    else:
        model_name = args.output

    try:
        train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs, "model_inter_" + model_name + ".pth")
    except Exception as e:
        traceback.print_exc()
        print(e)
    finally:
        print("saving model")
        torch.save(model.state_dict(), ("model_" + model_name + ".pth"))


if __name__ == '__main__':
    main()
