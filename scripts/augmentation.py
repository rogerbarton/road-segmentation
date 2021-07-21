import random
from PIL import Image
from glob import glob
import torchvision.transforms.functional as F
import argparse
import math
import os


def flip(image, mask):
    out = []
    # if random.random() < 0.7:
    out.append((F.hflip(image), F.hflip(mask)))
    # if random.random() < 0.7:
    #     out.append((F.vflip(image), F.vflip(mask)))
    return out


def rotate(image, mask):
    out = []
    for i in range(8):
        rnd = random.random()
        if rnd < 0.5:
            angle = random.randint(45 * i, 45 * (i + 1))
            image2 = F.rotate(image, angle)
            mask3 = F.rotate(mask, angle)
            out.append((image2, mask3))
            # out.append((F.rotate(image, angle, expand=True), F.rotate(mask, angle, expand=True)))
            # do a center crop and resize, in order to remove black borders
            # need to find biggest square
            """
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
            out.append((cropped, cropped_mask))
            """
    return out


def crop(image, mask, max_crop_factor=0.85):
    # image.show()
    # images are square anyway
    width_original = image.size[0]
    crop_width_min = int(width_original * max_crop_factor)
    # print(f"width: {width_original}, crop_min: {crop_width_min}")
    top = random.randint(0, width_original - crop_width_min)
    left = random.randint(0, width_original - crop_width_min)
    # height = random.randint(crop_width_min, min(width_original - top, width_original-left))
    height = random.randint(crop_width_min, width_original - top)
    width = random.randint(crop_width_min, width_original - left)
    # maybe not resize
    image_cropped = F.resized_crop(
        image, top=top, left=left, height=height, width=width, size=image.size
    )
    # XXX: maybe change interpolation for mask?
    mask_cropped = F.resized_crop(
        mask, top=top, left=left, height=height, width=width, size=image.size
    )
    return [(image_cropped, mask_cropped)]


def rotate_90(image, mask):
    out = []
    angles = [90, 180, 270]
    for a in angles:
        image_rot = F.rotate(image, a)
        mask_rot = F.rotate(mask, a)
        out.append((image_rot, mask_rot))
    return out


def invert(image, mask):
    if random.random() < 0.5:
        return [(F.invert(image), mask)]
    else:
        return []


def main():
    # todo different 78 neue groundtruth
    # todo different output paths
    fixed_groundtruths = [33, 41, 65, 78, 96, 99]
    with open("notes.txt") as f:
        harder_images = [int(val) for val in f.readlines()]
    harder_images = [str(val).zfill(3) for val in harder_images]
    harder_images = ["satImage_" + val + ".png" for val in harder_images]
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--training_path",
    #     help="path of the training folder, containing groundtruth and images",
    #     default="./training",
    # )
    parser.add_argument(
        "--clean", help="delete augmented data", default=False, action="store_true"
    )
    parser.add_argument(
        "--validation",
        help="augment the validation set",
        default=False,
        action="store_true",
    )
    parser.add_argument("--seed", help="fixed random seed", default=17, type=int)
    args = parser.parse_args()
    output_training = "training_augmented"
    output_validation = "validation_augmented"
    if not os.path.isdir(output_training):
        os.makedirs(f"{output_training}/images")
        os.makedirs(f"{output_training}/groundtruth")
    if not os.path.isdir(output_validation):
        os.makedirs(f"{output_validation}/images")
        os.makedirs(f"{output_validation}/groundtruth")
    source_dir = "./training"
    output_dir = output_training
    if args.validation:
        source_dir = "./validation"
        output_dir = output_validation
    if args.clean:
        to_delete = [x for x in glob("training/images/*.png") if "_aug" in x]
        to_delete += [x for x in glob("training/groundtruth/*.png") if "_aug" in x]
        to_delete += [x for x in glob("validation/images/*.png") if "_aug" in x]
        to_delete += [x for x in glob("validation/groundtruth/*.png") if "_aug" in x]
        to_delete += [x for x in glob(output_training + "/images/*.png")]
        to_delete += [x for x in glob(output_training + "/groundtruth/*.png")]
        to_delete += [x for x in glob(output_validation + "/images/*.png")]
        to_delete += [x for x in glob(output_validation + "/groundtruth/*.png")]
        for path in to_delete:
            os.remove(path)
        return

    random.seed(args.seed)
    original_images = [x for x in glob(source_dir + "/images/*.png") if "aug" not in x]
    harder_images = [source_dir + "/images/" + x for x in harder_images]
    for i, img_path in enumerate(original_images):
        harder = False
        if img_path in harder_images:
            print("processing harder image")
            harder = True
        print(f"processing image {i} of {len(original_images)}")
        image = Image.open(img_path)
        mask_path = img_path.replace("images", "groundtruth")
        mask = Image.open(mask_path)
        img_name = os.path.basename(img_path)[:-4]
        mask_name = os.path.basename(mask_path)[:-4]
        images_to_process = [(image, mask)]
        t = []
        # images_to_process.extend(invert(image, mask))
        for (i, m) in images_to_process:
            n_flips = 1
            if harder:
                n_flips = random.randint(1, 2)
            for _ in range(n_flips):
                t += flip(i, m)
        images_to_process += t
        t = []

        for (i, m) in images_to_process:
            n_rots = 1
            if harder:
                n_rots = random.randint(1, 2)
            for _ in range(n_rots):
                # t += rotate(i, m)
                t += rotate_90(i, m)
        images_to_process.extend(t)
        t = []

        """
        for (i, m) in images_to_process:
            n_crops = random.randint(0, 4)
            if harder:
                n_crops = random.randint(1, 6)
            for _ in range(n_crops):
                t += crop(i, m)
        images_to_process.extend(t)
        """
        print(len(images_to_process))
        for i, (img_aug, mask_aug) in enumerate(images_to_process):
            img_aug.save(f"{output_dir}/images/{img_name}_aug{i}.png")
            mask_aug.save(f"{output_dir}/groundtruth/{mask_name}_aug{i}.png")
    # noise
    # maybe occlusion


if __name__ == "__main__":
    main()
