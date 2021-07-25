echo "Train CNN Base\n\n\n\n\n"
python3 train.py --model=cnn --bs=128 --n_epochs=20

echo "Train Unet Base\n\n\n\n\n"
python3 train.py --model=unet --n_epochs=35

echo "Train New Unet Base\n\n\n\n\n"
python3 train.py --model=new_unet --n_epochs=35

echo "Train with altered images\n\n\n\n\n"
python3 train.py --model=new_unet --pre_processing=altered_images --n_epochs=80

echo "Train with altered images + otf\n\n\n\n\n"
python3 train.py --model=new_unet --pre_processing=otf --n_epochs=80

echo "Train with bigger kernel\n\n\n\n\n"
python3 train.py --model=new_unet --pre_processing=altered_images --n_epochs=80 --kernel_size=5

echo "Train with bigger kernel\n\n\n\n\n"
python3 train.py --model=new_unet --pre_processing=otf --n_epochs=80 --kernel_size=5

echo "Train with bigger kernel\n\n\n\n\n"
python3 train.py --model=new_unet --pre_processing=altered_images --n_epochs=200 --kernel_size=5

echo "Train with bigger kernel\n\n\n\n\n"
python3 train.py --model=new_unet --pre_processing=otf --n_epochs=200 --kernel_size=5

echo "Train with modified lr\n\n\n\n\n"
python3 train.py --model=new_unet --pre_processing=otf --kernel_size=3 --lr=0.0001 --n_epochs=200

echo "Train with modified ks=5 lr\n\n\n\n\n"
python3 train.py --model=new_unet --pre_processing=otf --kernel_size=5 --lr=0.0001 --n_epochs=200
