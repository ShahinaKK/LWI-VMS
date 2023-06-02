#!/bin/sh
DATASET_PATH=/home/shahina.kunhimon/PycharmProjects/unetr_plus_plus/DATASET
export SAVE_DIR=$1
export PYTHONPATH=./
export RESULTS_FOLDER=./"$SAVE_DIR"
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task06_Lung
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_lung 6 0 --shuffle_pretrain --mask_ratio 0.4 --mask_patch_size 4

export MODEL="${SAVE_DIR}"/unetr_pp/3d_fullres/Task006_Lung/unetr_pp_trainer_lung__unetr_pp_Plansv2.1/fold_0/model_final_checkpoint.model
export RESULTS_FOLDER=FINETUNE_"${SAVE_DIR}"
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task06_Lung
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw
python unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_lung 6 0 --pretrain_ckpt "$MODEL"

export CHECKPOINT_PATH= $RESULTS_FOLDER
export RESULTS_FOLDER=$RESULTS_FOLDER
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task06_Lung
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_lung 6 0  -val


