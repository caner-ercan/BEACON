from __future__ import print_function
import sys
#sys.path.insert(0, '/rsrch5/home/trans_mol_path/cercan/.conda/envs/clam/lib/python3.7/site-packages/')
import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import csv


def main(args):
    if args.multi_data_loaders:
        args.n_classes=2
        featureFolders = os.listdir(args.data_root_dir)
        dataset_multiple =[]
        for featureFolder in featureFolders:
            #the original image patches dataset
            if featureFolder == "feature_0":
                dataset_orig = Generic_MIL_Dataset(csv_path = args.csv_path,
                                data_dir= os.path.join(args.data_root_dir,"feature_0", "merged"),
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'diploid':0, 'aneuploid':1},
                                patient_strat=False,
                                ignore=[])
            else:
                dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                                        data_dir= os.path.join(args.data_root_dir,featureFolder,"merged"),
                                        shuffle = False, 
                                        seed = args.seed, 
                                        print_info = True,
                                        label_dict = {'diploid':0, 'aneuploid':1},
                                        patient_strat=False,
                                        ignore=[])
                dataset_multiple.append(dataset)
    elif args.merged_features:
        if args.task == 'task_1_tumor_vs_normal':
            args.n_classes=2
            dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                                    data_dir= os.path.join(args.data_root_dir,"merged"),
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'diploid':0, 'aneuploid':1},
                                    patient_strat=False,
                                    ignore=[])
            #the original image patches dataset
            dataset_orig = Generic_MIL_Dataset(csv_path = args.csv_path,
                                    data_dir= os.path.join(args.data_root_dir,"features_roi_orig", "merged"),
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'diploid':0, 'aneuploid':1},
                                    patient_strat=False,
                                    ignore=[])
        elif args.task == 'task_2_tumor_subtyping':
            args.n_classes=3
            dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                                    data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                                    patient_strat= False,
                                    ignore=[])

            if args.model_type in ['clam_sb', 'clam_mb', 'clam_sb_sc']:
                assert args.subtyping 
                
        else:
            raise NotImplementedError
    else:
        if args.task == 'task_1_tumor_vs_normal':
            args.n_classes=2
            dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                                    data_dir= args.data_root_dir,
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict = {'diploid':0, 'aneuploid':1},
                                    patient_strat=False,
                                    ignore=[])
            #the original image patches dataset
            dataset_orig = dataset
                

    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir,exist_ok=True)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_f1_score = []
    all__val_f1_score = []
    all_test_acc_0 = []
    all_test_acc_1 = []
    all_val_acc_0 = []
    all_val_acc_1 = []
    all_test_acc = []
    all_val_acc = []
    all_val_balanced_acc=[]
    all_test_balanced_acc=[]
    folds = np.arange(start, end)
    for i in folds:
        filename_pkl = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        if not os.path.isfile(filename_pkl):
            seed_torch(args.seed)
            if args.class_weight_1 == 0:
                reset_class_weight_1 = True
                args.class_weight_1 = getCalculateClassWeight('{}/splits_{}_descriptor.csv'.format(args.split_dir, i))
                print("calculated class 1 weight: "+ str(args.class_weight_1))  
            # create a datasetonly from the original image patches that you can use them for the validaton and test
            train_dataset_orig, val_dataset_orig, test_dataset_orig = dataset_orig.return_splits(from_id=False, 
                    csv_path='{}/splits_{}.csv'.format(args.split_dir, i)) 
            datasets_orig = (train_dataset_orig, val_dataset_orig, test_dataset_orig)
            
            if args.multi_data_loaders:
                from utils.core_utils_multipleDataloaders import train
                datasets = []
                for dataset in dataset_multiple:
                    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                                                                                    csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
                    datasets_single = (train_dataset, val_dataset, test_dataset)
                    datasets.append(datasets_single)
            else:
                from utils.core_utils import train
                train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                        csv_path='{}/splits_{}.csv'.format(args.split_dir, i)) 
                datasets = (train_dataset, val_dataset, test_dataset)
                

            results, test_auc, val_auc, test_acc, val_acc, val_accs, test_accs, test_f1_score, val_f1_score  = train(datasets, datasets_orig, i, args)
            all_test_auc.append(test_auc)
            all_val_auc.append(val_auc)
            all_test_f1_score.append(test_f1_score)
            all__val_f1_score.append(val_f1_score)
            all_test_acc_0.append(test_accs[0])
            all_test_acc_1.append(test_accs[1])
            all_val_acc_0.append(val_accs[0])
            all_val_acc_1.append(val_accs[1])
            all_val_acc.append(val_acc)
            all_test_acc.append(test_acc)
            all_val_balanced_acc.append(np.mean(val_accs))
            all_test_balanced_acc.append(np.mean(test_accs))
            #write results to pkl
            
            save_pkl(filename_pkl, results)

            # if reset_class_weight_1:
            #     args.class_weight_1 = 0

    final_df = pd.DataFrame({'folds': folds, 'int_test_auc': all_test_auc, 'val_auc': all_val_auc, 
                             'int_test_f1_score': all_test_f1_score,'val_f1_score': all__val_f1_score, 
                             'int_test_acc': all_test_acc, 'int_test_specificity': all_test_acc_0, 'int_test_sensitivity': all_test_acc_1, 
                             'val_acc' : all_val_acc, 'val_specificity': all_val_acc_0, 'val_sensitivity': all_val_acc_1,
                             'int_test_balanced_acc': all_test_balanced_acc,'val_balanced_acc' : all_val_balanced_acc})
    
    final_df = final_df.round(3)
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    




# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory, features top dir')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enable dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'fce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'clam_sb_sc'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', 'fce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
# CE added options
parser.add_argument('--csv_path', type=str, help='csv path')
parser.add_argument('--stop_epoch', type=int, default=40, help='stop_epoch')
parser.add_argument('--schedul_pat', type=int, default=8, help='schedul_patience')
parser.add_argument('--schedul_cool', type=int, default=10, help='schedul_cooldown')
parser.add_argument('--schedul_factor', type=float, default=0.5, help='schedul_factor')
parser.add_argument('--class_weight_1', type=float, help='CO weight')
parser.add_argument('--two_rounds', action='store_true', default=False, help='two rounds trainings with and without class_weights')
parser.add_argument('--multi_data_loaders', action='store_true', default=False, help='multiple dataloader with different augmentations')
parser.add_argument('--merged_features', action='store_true', default=False, help='the augmented and original features are stored in the same folder structure seperately. The merged ones will be used for the training and the original ones for the rest. If multi_data_loaders is activated, this is obsolete.')
parser.add_argument('--arch', type=str, default=None, help='name of the backbone. Only for the summary file')
parser.add_argument('--first_layer_attention', action='store_true', default=False)
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


# if args.two_rounds and args.class_weight_1 !=1:
#     args.round=0
# else:
#     args.round=2


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)
if args.arch is None:
    args.arch= os.path.basename(args.data_root_dir)
encoding_size = 1024
settings = {'backbone': args.arch,
            'num_splits': args.k, 
            'experiment': args.exp_code,
            'bag_loss': args.bag_loss,
            'model_type': args.model_type,
            'model_size': args.model_size,
            'model_first_layer_attention': args.first_layer_attention,
            'class_weight_1': args.class_weight_1,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'reg': args.reg,
            'label_frac': args.label_frac,
            'seed': args.seed,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt,
            'csv_path': args.csv_path,
            'schedul_pat': args.schedul_pat,
            'schedul_cool': args.schedul_cool,
            'schedul_factor': args.schedul_factor,
            'lr': args.lr}

if args.model_type in ['clam_sb', 'clam_mb', 'clam_sb_sc']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

    
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir,exist_ok=True)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir,exist_ok=True)

if args.split_dir is None:
    args.split_dir = os.path.join(os.path.dirname(args.results_dir),'splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)



settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()
print(settings)
csv_settings_path=os.path.join(args.results_dir,'settings.csv')
with open(csv_settings_path, 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=settings.keys())
    writer.writeheader()
    writer.writerow(settings)



# while args.round <3:
#     if args.round==1:
#         print("updating settings for the second round")
#         args.old_results_dir=args.results_dir   
#         args.results_dir=os.path.join(args.results_dir+'_secondRound')
#         args.class_weight_1=float(1)    
#         args.lr=args.lr/10
print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")
    # args.round = args.round*2+1


