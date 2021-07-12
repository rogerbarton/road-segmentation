from train import UNet
import torch
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import argparse
from glob import glob
import numpy as np
import cv2
from train import load_all_from_path, np_to_tensor
import re


# some constants
PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road

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

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()

    parser.add_argument('model', metavar='model', type=str, nargs=1, help='the model')
    args = parser.parse_args()

    print(args.model)
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.model[0]))
    model.eval()

    # predict on test set
    test_path = 'test'
    test_filenames = (glob(test_path + '/*.png'))
    test_images = load_all_from_path(test_path)
    # batch_size = test_images.shape[0]
    size = test_images.shape[1:3]
    # we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
    test_images = np.stack([cv2.resize(img, dsize=(384, 384)) for img in test_images], 0)
    test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)

    test_pred = [model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]


    test_pred = np.concatenate(test_pred, 0)
    test_pred = np.moveaxis(test_pred, 1, -1)  # CHW to HWC
    test_pred = np.stack([cv2.resize(img, dsize=size) for img in test_pred], 0)  # resize to original shape
    
    # morphological postprocessing
    
    kernel = np.ones((3, 3), np.uint8)
    test_pred = np.stack([cv2.erode(img, kernel, iterations=7) for img in test_pred], 0)
    test_pred = np.stack([cv2.dilate(img, kernel, iterations=7) for img in test_pred], 0)
    
    # now compute labels
    test_pred = test_pred.reshape((-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE))
    test_pred = np.moveaxis(test_pred, 2, 3)
    test_pred = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF)
    # create_submission(test_pred, test_filenames, submission_filename='unet_submission.csv')

    #test_pred = morphological_postprocessing(test_pred)

    create_submission(test_pred, test_filenames, submission_filename='unet_postprocessed_submission.csv')


if __name__ == '__main__':
    main()
