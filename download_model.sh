#!/bin/bash
set -e

## ==============================================================================================================
## Description   : Downloads the trained taskboard NN detection model.
##                 NOTE: Only invoke from within the tum-tb-perception directory!
## ==============================================================================================================

model_dir_path="./models"
model_url="https://syncandshare.lrz.de/dl/fi8T5K7jJcUw2kKS1NL94j/tb_fasterrcnn_epochs_25_batches_1_tv_ratio_07_seed_2_20240121_154144.pt"
model_file_name="tb_fasterrcnn_epochs_25_batches_1_tv_ratio_07_seed_2_20240121_154144.pt"

# Set up directory:
if [ ! -d ${model_dir_path} ]; then
    echo "[download_model] [INFO]: Directory ${model_dir_path} does not exist. Creating now..."
    mkdir -p ${model_dir_path};
fi

# Download model:
wget -O "${model_dir_path}/${model_file_name}" "${model_url}"
