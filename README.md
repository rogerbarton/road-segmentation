# Road Segmentation for Computer Intelligence Lab

## installation
The packages needed are in requirements.txt, install them using pip:

```bash
pip install -r requirements.txt
```
## Preprocessing
#### Artificial Road Addition
Since the dataset is only 100 images with groundtruths, we explored the option of adding random road cutouts to the existing satellite images, in order to create new street layouts. These new images are saved in the `traingin_occluded` folder, which will be used by the `augmentation.py` script, and also roatated and flipped. The `--amount` can be used to specify how many times a single road should be added to the original picture. (i.e. `occlusion.py --amount 2` would create 2 additional image from each of the originals, increasing the dataset by 180 images). The `augmentation.py` script has to be run after `occlusion.py`.

``` bash
python scripts/occlusion.py --amount 1
```

#### Flip and Rotation
In order to have more training data, we flip each image and rotate both versions by 90, 180 and 270 degrees. This generates 7 additional images to each base image. We also do the same with the validation split of the dataset. These images get saved in the `training_augmented` and `validation_augmented` folders respectively, which are used as source for the `train.py` script. The augmented images can be deleted using the `--clean` command line argument. (see `--help` for all supported arguments)

``` bash
python scripts/augmentation.py
python scripts/augmnetation.py --validation
```

## Training
The training is done using the `train.py` script, which has the following command line arguments:

| Argument | Description |
| --- | --- |
| -h | Show help |
| --model {`cnn`, `unet`, `new_unet`} | Which model to train, default: `new_unet`|
| --kernel_size {`3`, `5`, `7`} | Choose the size of the kernels for `new_unet`, default: `3` |
| --pre_processing {`otf`, `none`, `altered_images`} | What preprocessing technique to use, `otf` adds random crop, rotation, shearing and color augmentation in the dataloader, `altered_images` uses the preprocessing from the previous step only and `none` uses the original training data without any preprocessing. For `otf` and `altered_images`, the `augmentation.py` scripts has to be used to generate the images., default: `none` |
| --loss {`BCE`, `BCE_with_logit`} | Which loss function to use, BCELoss or BCEWithLogitsLoss, default: `BCE` |
| --lr `LR` | What learining rate to use, default: `0.001` |
| --optimizer {`adam`, `sgd`, `adamax`} | What optimizer to use, default: `adam` |
| --output `OUTPUT` | Name of the output file |
| --n_epochs `EPOCHS` | How many epochs to train, default: `40` |
| --seed `SEED` | Fixed random seed, default: `17` |
| --bs `BS` | What batch size to use, default: `4` |

The training script produces the final weights as an output, by default called `"model_new_unet_....pth"` as well as the weights of the epoch with the highest validation accuracy, called `"model_inter_new_unet_....pth"`. These files can be used as input for the `post_process.py` script

## Post Processing
To apply our post- processing methods, use the provided post- processing script, called `post_process.py`. To run this script, call it with the following command, where `model` is the model you wish to apply post processing to:

```bash
python3 post_process.py model [parameters]
```

The following list contains all possible command line arguments:

| Argument | Description |
| --- | --- |
| --model {`unet`, `new_unet`} | Which type model to train, default: `new_unet`|
| --kernel_size {`3`, `5`, `7`} | Choose the size of the kernels for `new_unet`, default: `3` |
| --post_process {`Morph4`, `Morph7`, `CRF`, `All`, `None`} | Which post-processing approach to use, default: `All` |
| --porcess_one {`True`, `False`} | Whether only one image should be processed, for testing purposes, default: `False`|

## Kaggle submission
Our kaggle submission was trained with the parameters and preprocessing in the `run_kaggle.sh` script, which can be used to replicate our score. It will generate a file called `prediction.csv` which is the right format.
