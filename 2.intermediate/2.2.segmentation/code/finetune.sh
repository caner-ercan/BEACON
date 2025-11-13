#!/bin/bash

# Published setting: MiT-B0, lr 1e-4, fine-tuned from the TCGA-pretrained checkpoint.
: "${BE_MASTER:?BE_MASTER environment variable must be set to the data root}"

model_name="nvidia/mit-b0"
weight_path="${BE_MASTER}/2.intermediate/2.3.segmentation/training/finetuning/tcga_pretrain/mit-b0_tcga/0.0001"
lr=1e-4

echo "Model: $model_name"
echo " Model weights: $weight_path"
echo "Learning Rate: $lr"

python "${BE_MASTER}/2.intermediate/2.3.segmentation/finetuner/code/eval.py" --model_name $model_name --lr $lr --weight_path $weight_path