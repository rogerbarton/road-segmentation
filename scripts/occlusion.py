import random
from PIL import Image
from glob import glob
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision.transforms.functional as F
import cv2


cutouts = None


def extract_roads(img, mask):
    img = np.array(img)
    mask = np.array(mask)
    kernel = np.ones((5, 5), np.uint8)
    # mask_dilated = cv2.dilate(mask, kernel, iterations=3)
    # print(np.unique(mask, return_counts=True))
    img[mask == 0] = 0
    # plt.imshow(img)
    # plt.show()
    return Image.fromarray(img)


def init_cutouts(cutout_paths, mask_paths):
    global cutouts
    if cutouts is not None:
        return
    cutouts = []
    for i, cutout in enumerate(cutout_paths):
        cutouts.append(
            (
                Image.open(cutout),
                Image.open(mask_paths[i]),
                "vaug" in cutout,
            )
        )


def main():
    fixed_groundtruths = ["033", "041", "065", "078", "096", "099"]
    global cutouts
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_path",
        help="path of the training folder, containing groundtruth and images",
        default="./training",
    )
    parser.add_argument(
        "--amount",
        help="how many times to add a road to each image",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--clean", help="delete augmented data", default=False, action="store_true"
    )
    parser.add_argument("--seed", help="fixed random seed", default=17, type=int)
    args = parser.parse_args()
    random.seed(args.seed)

    output_training = "training_occluded"
    if not os.path.isdir(output_training):
        os.makedirs(f"{output_training}/images")
        os.makedirs(f"{output_training}/groundtruth")
    if args.clean:
        augmented_images = [
            x for x in glob(args.training_path + "/images/*.png") if "_occl" in x
        ]
        augmented_masks = [
            x for x in glob(args.training_path + "/groundtruth/*.png") if "_occl" in x
        ]
        to_delete = [x for x in glob(output_training + "/images/*.png")]
        to_delete += [x for x in glob(output_training + "/groundtruth/*.png")]
        for path in augmented_images + augmented_masks + to_delete:
            os.remove(path)
        return
    original_images = [
        x
        for x in glob(args.training_path + "/images/*.png")
        if "_occl" not in x and "_aug" not in x
    ]
    cutout_paths = [x for x in sorted(glob(args.training_path + "/augment/*.png"))]
    cutout_masks = [
        x for x in sorted(glob(args.training_path + "/augment/masks/*.png"))
    ]
    init_cutouts(cutout_paths, cutout_masks)

    # visually inspect cutouts for correctness
    # for i, cutout in enumerate(cutouts):
    #     c = Image.open(cutout)
    #     m = Image.open(cutout_masks[i])
    #     _, axarr = plt.subplots(1, 2)
    #     axarr[0].imshow(c)
    #     axarr[1].imshow(m)
    #     plt.show()

    for i, img_path in enumerate(original_images):
        print(f"processing image {i} of {len(original_images)}")
        image = np.array(Image.open(img_path))
        mask_path = img_path.replace("images", "groundtruth")
        if any(x in img_path for x in fixed_groundtruths):
            print(f"use fixed ground truth for image {img_path}")
            mask_path = img_path.replace("images", "fixed_groundtruth")
        mask = np.array(Image.open(mask_path))
        img_name = os.path.basename(img_path)[:-4]
        mask_name = os.path.basename(mask_path)[:-4]
        for j in range(args.amount):
            c, m, is_vertical = random.choice(cutouts)
            # rot = random.randint(-10, 10)
            # shift image (left/right if vertical)
            # shifty = random.randint(-50, 50)
            # shiftx = random.randint(-50, 50)
            # c = F.affine(c, angle=rot, translate=(shiftx, shifty), scale=1, shear=0)
            # m = F.affine(m, angle=rot, translate=(shiftx, shifty), scale=1, shear=0)
            shift = random.randint(-50, 50)
            rot = 0
            if is_vertical:
                # shift road a little bit left or right
                c = F.affine(c, angle=rot, translate=(shift, 0), scale=1, shear=0)
                m = F.affine(m, angle=rot, translate=(shift, 0), scale=1, shear=0)
            else:
                # shift up/down
                c = F.affine(c, angle=rot, translate=(0, shift), scale=1, shear=0)
                m = F.affine(m, angle=rot, translate=(0, shift), scale=1, shear=0)

            c = np.array(c)
            m = np.array(m)
            image_aug = image.copy()
            image_aug[c != 0] = 0
            image_aug = image_aug + c
            aug_mask = (
                (mask.astype(np.uint16) + m.astype(np.uint16))
                .clip(0, 237)
                .astype(np.uint8)
            )
            # _, axarr = plt.subplots(2, 2)
            # axarr[0, 0].imshow(image)
            # axarr[0, 1].imshow(aug_mask)
            # axarr[1, 0].imshow(c)
            # axarr[1, 1].imshow(m)
            # plt.show()
            # extracted = extract_roads(image, mask)
            # extracted.save(img.replace("images", "extracted_roads"))
            image_aug = Image.fromarray(image_aug)
            aug_mask = Image.fromarray(aug_mask)
            image_aug.save(f"{output_training}/images/{img_name}_occl{j}.png")
            aug_mask.save(f"{output_training}/groundtruth/{mask_name}_occl{j}.png")


if __name__ == "__main__":
    main()
