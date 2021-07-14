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
    global cutouts
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_path",
        help="path of the training folder, containing groundtruth and images",
        default="./training",
    )
    parser.add_argument(
        "--clean", help="delete augmented data", default=False, action="store_true"
    )
    parser.add_argument("--seed", help="fixed random seed", default=17, type=int)
    args = parser.parse_args()
    random.seed(args.seed)
    if args.clean:
        augmented_images = [
            x for x in glob(args.training_path + "/images/*.png") if "_occl" in x
        ]
        augmented_masks = [
            x for x in glob(args.training_path + "/groundtruth/*.png") if "_occl" in x
        ]
        for path in augmented_images + augmented_masks:
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

    for i, img in enumerate(original_images):
        print(f"processing image {i} of {len(original_images)}")
        image = np.array(Image.open(img))
        mask_path = img.replace("images", "groundtruth")
        mask = np.array(Image.open(mask_path))
        img_name = img[:-4]
        mask_name = mask_path[:-4]
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
        image[c != 0] = 0
        image = image + c
        aug_mask = (
            (mask.astype(np.uint16) + m.astype(np.uint16)).clip(0, 237).astype(np.uint8)
        )
        # _, axarr = plt.subplots(2, 2)
        # axarr[0, 0].imshow(image)
        # axarr[0, 1].imshow(aug_mask)
        # axarr[1, 0].imshow(c)
        # axarr[1, 1].imshow(m)
        # plt.show()
        # extracted = extract_roads(image, mask)
        # extracted.save(img.replace("images", "extracted_roads"))
        image = Image.fromarray(image)
        aug_mask = Image.fromarray(aug_mask)
        image.save(f"{img_name}_occl.png")
        aug_mask.save(f"{mask_name}_occl.png")


if __name__ == "__main__":
    main()
