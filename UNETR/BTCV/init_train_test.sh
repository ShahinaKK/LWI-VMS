#!/bin/sh
export SAVE_DIR=$1
python main.py  --logdir $SAVE_DIR --save_checkpoint --val_every 100 --max_epochs 800 --shufflemask_pretrain --mask_ratio 0.4 --mask_patch_size 16 --data_dir /path/to/dataset --json_list dataset_0.json

export DIR=./runs/$SAVE_DIR
export NEW_DIR=FINETUNE_$SAVE_DIR
python main.py --logdir $NEW_DIR --save_checkpoint --val_every 100 --max_epochs 5000 --finetune_pretrained --resume_ckpt --pretrained_dir $DIR  --pretrained_model_name model_final.pt --data_dir ./path/to/dataset --json_list dataset_0.json

python test.py --pretrained_dir ./runs/$NEW_DIR --pretrained_model_name model.pt --infer_overlap 0.5 --json_list dataset.json --data_dir ./path/to/dataset --json_list dataset_0.json
