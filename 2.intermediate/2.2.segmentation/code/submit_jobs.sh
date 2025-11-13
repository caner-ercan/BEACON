#!/bin/bash

: "${BE_MASTER:?BE_MASTER environment variable must be set to the data root}"

cd "${BE_MASTER}/2.intermediate/2.3.segmentation/training/finetuning/from_tcga_ds1"
job-runner.sh "${BE_MASTER}/2.intermediate/2.3.segmentation/finetuner/code/k8_yamls/job_submit.yaml"