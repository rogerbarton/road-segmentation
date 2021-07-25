# road-segmentation
Road Segmentation for CIL

## Preprocessing
#### Artificial Road addition
Since the dataset is only 100 images with groundtruths, we explored the option of adding random road cutouts to the existing satellite images, in order to create new street layouts. These new images are saved in the `traingin_occluded` folder, which will be used by the `augmentation.py` script, and also roatated and flipped. The `--amount` can be used to specify how many times a single road should be added to the original picture. (i.e. `occlusion.py --amount 2` would create 2 additional image from each of the originals, increasing the dataset by 180 images). The `augmentation.py` script has to be run after `occlusion.py`.

``` bash
python scripts/occlusion.py --amount 1
```

#### flip and rotation
In order to have more training data, we flip each image and rotate both versions by 90, 180 and 270 degrees. This generates 7 additional images to each base image. We also do the same with the validation split of the dataset. These images get saved in the `training_augmented` and `validation_augmented` folders respectively, which are used as source for the `train.py` script. The augmented images can be deleted using the `--clean` command line argument. (see `--help` for all supported arguments)

``` bash
python scripts/augmentation.py
python scripts/augmnetation.py --validation
```

## training

## Post Processing
