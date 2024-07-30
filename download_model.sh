#!/bin/bash
set -e

## ==============================================================================================================
## Description   : Downloads the trained taskboard NN detection model.
##                 NOTE: Only invoke from within the tum-tb-perception directory!
## ==============================================================================================================

models_dir_path="./models"
model_url="https://drive.usercontent.google.com/download?id=1bY65rgKtC6jfZ8He2Nt84YNv5JDvs0D2&confirm=xxx"
model_file_name="tb_fasterrcnn_epochs_25_batches_1_tv_ratio_07_seed_2_20240121_154144.pt"

# Set up directory:
if [ ! -d ${model_dir_path}/ ]; then
    mkdir -p ${model_dir_path}/;
fi

# Download model:
cd ${models_dir_path}
curl "${model_url}" -o "${model_file_name}"
cd ..
