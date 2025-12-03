#!/bin/bash

# get first argument

exp_no=$1 # 0-14

model_names=("nvidia/mit-b0" "nvidia/mit-b1" "nvidia/mit-b2" "nvidia/mit-b3" "nvidia/mit-b4" "nvidia/mit-b5")
pretrained_weight_paths=(
    "/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/training/finetuning/tcga_pretrain/mit-b0_tcga/0.0001" \
    "/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/training/finetuning/tcga_pretrain/mit-b1_tcga/0.0001" \
    "/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/training/finetuning/tcga_pretrain/mit-b2_tcga/0.001" \
    "/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/training/finetuning/tcga_pretrain/mit-b3_tcga/0.0001" \
    "/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/training/finetuning/tcga_pretrain/mit-b4_tcga/0.001" \
    "/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/training/finetuning/tcga_pretrain/mit-b5_tcga/0.001")


lr_options=(1e-3 1e-4 1e-5)


model_index=$((exp_no %       ${#model_names[@]}))
lr_index=$(( (exp_no /     ${#model_names[@]}) % ${#lr_options[@]} ))


model_name=${model_names[$model_index]}
weight_path=${pretrained_weight_paths[$model_index]}
lr=${lr_options[$lr_index]}


echo "Model: $model_name"
echo " Model weights: $weight_path"
echo "Learning Rate: $lr"

python /rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/finetuner/code/eval.py --model_name $model_name --lr $lr --weight_path $weight_path 