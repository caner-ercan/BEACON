#!/bin/bash


code_dir="/rsrch5/home/trans_mol_path/cercan/code/CellViT-plus-plus/cellvit/train_cell_classifier_head.py"
finetuning_folder="/rsrch5/home/trans_mol_path/cercan/data/BE/cell_detect/cellvit/250501_finetuning"
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