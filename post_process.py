import torch
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import argparse
from glob import glob
import numpy as np
import cv2
from train import np_to_tensor
import re
import matplotlib.pyplot as plt
from PIL import Image

import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

# uncomment for old unet
# from train import UNet
from models.u_net import UNet

# some constants
PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road


def load_all_from_path(path):
    # loads all HxW .pngs contained in path as a 4D np.array of shape (n_images, H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    return np.stack([np.array(Image.open(f)) for f in sorted(glob(path + '/*.png'))]).astype(np.float32) / 255.

# def load_all_from_path_resize(path, resize_to=(400, 400)):
#     return np.stack([cv2.resize(np.array(Image.open(f)), dsize=resize_to) for f in sorted(glob(path + '/*.png'))]).astype(np.float32) / 255.


def morphological_postprocessing(imgs):
    out = []
    for img in imgs:
        kernel = np.ones((3,3), np.uint8)
        img = np.float32(img)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        out.append(img)
    return out

def post_processing(outs):
    return outs

def create_submission(test_pred, test_filenames, submission_filename):
    test_path='test_images/test_images'
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn, patch_array in zip(sorted(test_filenames), test_pred):
            img_number = int(re.search(r"\d+", fn).group(0))
            for i in range(patch_array.shape[0]):
                for j in range(patch_array.shape[1]):
                    f.write("{:03d}_{}_{},{}\n".format(img_number, j*PATCH_SIZE, i*PATCH_SIZE, int(patch_array[i, j])))


class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        #d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        # d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        # #d.addPairwiseBilateral(
        # #    sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        # #)
        # d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=image, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        d.addPairwiseGaussian(sxy=(0.05,0.05), compat=15, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        d.addPairwiseBilateral(sxy=(120,120), srgb=(40,40,40), rgbim=image, compat=8, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()

    parser.add_argument('model', metavar='model', type=str, nargs=1, help='the model')
    args = parser.parse_args()

    print(args.model)
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.model[0], map_location=device))
    model.eval()

    # predict on test set
    test_path = 'test'
    test_filenames = (glob(test_path + '/*.png'))
    test_images = load_all_from_path(test_path)
    #test_images = np.array([test_images[0], test_images[1]])
    # batch_size = test_images.shape[0]
    size = test_images.shape[1:3]
    # we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
    test_images = np.stack([cv2.resize(img, dsize=(384, 384)) for img in test_images], 0)
    test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)

    test_pred = [model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]


    test_pred = np.concatenate(test_pred, 0)
    test_pred = np.moveaxis(test_pred, 1, -1)  # CHW to HWC
    test_pred = np.stack([cv2.resize(img, dsize=size) for img in test_pred], 0)  # resize to original shape

    post_processor = DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=3,   # 3
        pos_w=3,        # 3
        bi_xy_std=140,  # 121, 140
        bi_rgb_std=5,   # 5, 5
        bi_w=5,         # 4, 5
    )

    raw_images = np.stack([np.array(Image.open(f)) for f in sorted(glob(test_path + '/*.png'))])

    inv_func = np.vectorize(lambda x : 1 - x)

    output = []

    for i, pred in enumerate(test_pred):
        print("Processing image " + str(i))
        raw_img = raw_images[i]
        raw_img = raw_img.astype(np.uint8)
        plt.imsave('raw.jpeg', raw_img)
        plt.imsave('first.jpeg', pred)
        plt.imsave('sec.jpeg', inv_func(pred))
        probs_labels = np.array([inv_func(pred), pred])
        prob = post_processor(raw_img, probs_labels)
        output.append(np.argmax(prob, axis=0).astype(np.uint8))
        plt.imsave('post_proc.jpeg', output[i])

    test_pred = np.array(output)

    

    """
    # morphological postprocessing
    
    kernel = np.ones((3, 3), np.uint8)
    test_pred = np.stack([cv2.erode(img, kernel, iterations=7) for img in test_pred], 0)
    test_pred = np.stack([cv2.dilate(img, kernel, iterations=7) for img in test_pred], 0)
    """
    
    # now compute labels
    test_pred = test_pred.reshape((-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE))
    test_pred = np.moveaxis(test_pred, 2, 3)
    test_pred = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF)
    # create_submission(test_pred, test_filenames, submission_filename='unet_submission.csv')

    #test_pred = morphological_postprocessing(test_pred)

    create_submission(test_pred, test_filenames, submission_filename='prediction.csv')


if __name__ == '__main__':
    main()
