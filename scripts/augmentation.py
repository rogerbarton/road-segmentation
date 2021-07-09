import random
from PIL import Image
from glob import glob
import torchvision.transforms.functional as F
import argparse
import math


def flip(path):
    image = Image.open(path)
    mask_path = path.replace("images", "groundtruth")
    mask = Image.open(mask_path)
    img_name = path[:-4]
    rnd = random.random()
    # if rnd <= 0.5:
    img_flip = F.hflip(image)
    mask_flip = F.hflip(mask)
    img_flip_name = img_name + "hflip" + ".png"
    img_flip.save(img_flip_name)
    maskstr = img_flip_name.replace("images", "groundtruth")
    mask_flip.save(maskstr)
    # else:
    img_flip = F.vflip(image)
    mask_flip = F.hflip(mask)
    img_flip_name = img_name + "vflip" + ".png"
    img_flip.save(img_flip_name)
    maskstr = img_flip_name.replace("images", "groundtruth")
    mask_flip.save(maskstr)


def rotate(path):
    image = Image.open(path)
    mask = path.replace("images", "groundtruth")
    mask2 = Image.open(mask)
    img2 = path[:-4]
    for i in range(8):
        rnd = random.random()
        if rnd < 1.0:
            angle = random.randint(45 * i, 45 * (i + 1))
            image2 = F.rotate(image, angle, expand=True)
            mask3 = F.rotate(mask2, angle, expand=True)
            # do a center crop and resize, in order to remove black borders
            # need to find biggest square

            length = int(
                image.size[0]
                / (
                    abs(math.cos(math.radians(angle)))
                    + abs(math.sin(math.radians(angle)))
                )
            )
            cropped = F.center_crop(image2, [length, length])
            cropped = F.resize(cropped, image.size)
            cropped_mask = F.center_crop(mask3, [length, length])
            cropped_mask = F.resize(cropped_mask, image.size)
            img2 = path[:-4]
            img2 = img2 + "_rot" + str(angle) + ".png"
            cropped.save(img2)
            maskstr = img2.replace("images", "groundtruth")
            cropped_mask.save(maskstr)


def crop(path, max_crop_factor=2):
    image = Image.open(path)
    mask_path = path.replace("images", "groundtruth")
    mask = Image.open(mask_path)
    img_name = path[:-4]
    mask_name = mask_path[:-4]
    # image.show()
    # images are square anyway
    width_original = image.size[0]
    top = random.randint(0, width_original // max_crop_factor)
    left = random.randint(0, width_original // max_crop_factor)
    # height = random.randint(width_original // max_crop_factor, min(width_original - top, width_original-left))
    height = random.randint(width_original // max_crop_factor, width_original - top)
    width = random.randint(width_original // max_crop_factor, width_original - left)
    # maybe not resize
    image_cropped = F.resized_crop(
        image, top=top, left=left, height=height, width=width, size=image.size
    )
    # XXX: maybe change interpolation for mask?
    mask_cropped = F.resized_crop(
        mask, top=top, left=left, height=height, width=width, size=image.size
    )
    # print(f"_cropped_{top}_{left}_{height}x{width}.png")
    image_cropped.save(img_name + f"_cropped_{top}_{left}_{height}x{width}.png")
    mask_cropped.save(mask_name + f"_cropped_{top}_{left}_{height}x{width}.png")
    # image_cropped.show()
    # mask_cropped.show()


def rotate_90(path):
    image = Image.open(path)
    mask_path = path.replace("images", "groundtruth")
    mask = Image.open(mask_path)
    img_name = path[:-4]
    mask_name = mask_path[:-4]
    # angle = random.choice([90,180,270])
    angles = [90, 180, 270]
    for a in angles:
        image_rot = F.rotate(image, a)
        mask_rot = F.rotate(mask, a)
        image_rot.save(img_name + "_rot" + str(a) + ".png")
        mask_rot.save(mask_name + "_rot" + str(a) + ".png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_path",
        help="path of the training folder, containing groundtruth and images",
        default="./training",
    )
    parser.add_argument("--seed", help="fixed random seed", default=17, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    for i, img in enumerate(glob(args.training_path + "/images/*.png")):
        print(f"augmentint picture {i}")
        flip(img)
        rotate(img)
    for i, img in enumerate(glob(args.training_path + "/images/*.png")):
        # random crops on images made so far
        print(f"cropping picture {i}")
        for _ in range(4):
            crop(img)
        # invert
        # noise
        # combine
        # maybe occlusion
        # squish


if __name__ == "__main__":
    main()
