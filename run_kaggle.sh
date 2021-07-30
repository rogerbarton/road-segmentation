# run the artificial occlusion script
python3 scripts/occlusion.py

# run the augmentation script
touch notes.txt
python3 scripts/augmentation.py
python3 scripts/augmentation.py --validation

# train the model
python3 train.py --model=new_unet --pre_processing=otf --n_epochs=400

# post_process the model
python3 post_process.py model_inter_new_unet_400_lr=0.001_BCE_otf_ks=3.pth --post_process=All

# the output file is called submission.csv
echo "Done"
