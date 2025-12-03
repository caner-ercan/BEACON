#!/bin/bash

# Log file
# cd /rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/training/finetuning
cd /rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/training/finetuning/from_tcga_ds1
for exp_no in {0..17}; do
  job-runner.sh /rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/finetuner/code/k8_yamls/job_submit_$exp_no.yaml
done