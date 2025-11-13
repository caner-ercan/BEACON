#!/bin/bash

: "${BE_MASTER:?BE_MASTER environment variable must be set to the data root}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_dir="${script_dir}/../cellvit/train_cell_classifier_head.py"
finetuning_folder="${BE_MASTER}/data/BE/cell_detect/cellvit/250501_finetuning"
cd $finetuning_folder

#get_folder list for finetuning
exp_list=("vanilla" "consep" "lizard" "panoptils" "ocelot" "midog" "nucls_main" "nucls_super")
# loop exp_list
for exp in "${exp_list[@]}"; do
    exp_folder="$finetuning_folder/$exp"
    cd $exp_folder
    config_file="$exp_folder/config_sweep.yaml"

    python $code_dir --config $config_file --sweep
done