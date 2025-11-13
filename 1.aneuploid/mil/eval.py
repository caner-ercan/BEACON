from __future__ import print_function
import sys
import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
# import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--fold', type=int, default=0, help='index of the single split/checkpoint to evaluate (default: 0)')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
parser.add_argument('--save_dir', type=str)
parser.add_argument('--csv_dir', type=str)
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = args.csv_dir,
                            #data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'diploid':0, 'aneuploid':1},
                            # label_dict = {'NCO':0, 'CO':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

# elif args.task == 'tcga_kidney_cv':
#     args.n_classes=3
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_kidney_clean.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'tcga_kidney_20x_features'),
#                             shuffle = False, 
#                             print_info = True,
#                             label_dict = {'TCGA-KICH':0, 'TCGA-KIRC':1, 'TCGA-KIRP':2},
#                             patient_strat= False,
#                             ignore=['TCGA-SARC'])

else:
    raise NotImplementedError

# Single fixed split/checkpoint, identified by args.fold (matches s_{fold}_checkpoint.pt) —
# one evaluation run, no fold sweep.
ckpt_path = os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(args.fold))
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    if datasets_id[args.split] < 0:
        split_dataset = dataset
    else:
        csv_path = '{}/splits_{}.csv'.format(args.splits_dir, args.fold)
        datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
        split_dataset = datasets[datasets_id[args.split]]
    args.writer_dir = os.path.join(args.save_dir, str(args.fold))
    model, patient_results, test_error, auc, df, accs, f1_score  = eval(split_dataset, args, ckpt_path)
    df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(args.fold)), index=False)

    summary_df = pd.DataFrame({'fold': [args.fold], 'test_auc': [auc], 'test_overall_acc': [1 - test_error],
                                'test_specificity': [accs[0]], 'test_sensitivity': [accs[1]],
                                'test_balanced_acc': [np.mean(accs)]})
    summary_df.to_csv(os.path.join(args.save_dir, 'summary.csv'))