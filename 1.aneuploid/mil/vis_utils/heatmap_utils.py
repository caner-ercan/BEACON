import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from utils.utils import *
from PIL import Image
from math import floor
import matplotlib.pyplot as plt
from datasets.wsi_dataset import Wsi_Region
import h5py
from wsi_core.WholeSlideImage import WholeSlideImage
from scipy.stats import percentileofscore
import math
from utils.file_utils import save_hdf5
from scipy.stats import percentileofscore

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile

def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level = -1, thresh=0.5, attn_scores = None, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)
    
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(2)
    
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, thresh=thresh, attn_scores = attn_scores, **kwargs)        
    return heatmap

def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object

# class FeatureDataset(Dataset):
#     def __init__(self, h5_file_path, pt_file_path, transform=None):
#         self.h5_file_path = h5_file_path
#         self.pt_file_path = pt_file_path
#         #self.transform = transform

#         with h5py.File(h5_file_path, 'r') as h5_file:
#             self.coords = h5_file['coords'][:]
#             self.features = h5_file['features'][:]

#         # elif file_path.endswith('.pt'):
#         #     self.features = torch.load(file_path)

#     def __len__(self):
#         return len(self.coords)

#     def __getitem__(self, idx):
#         coord = self.coords[idx]
#         feature = self.features[idx]
#         feature = torch.from_numpy(feature)

#         return feature, coord


def compute_from_features(wsi_object, clam_pred=None, model=None, features_path=None, batch_size=512,  
    attn_save_path=None, ref_scores=None, tile_prob=False, **wsi_kwargs):      
    # top_left = wsi_kwargs['top_left']
    # bot_right = wsi_kwargs['bot_right']
    # patch_size = wsi_kwargs['patch_size']
    
    # roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=8)
    # print('total number of patches to process: ', len(roi_dataset))
    # num_batches = len(roi_loader)
    # print('number of batches: ', len(roi_loader))
    mode = "w"

    hdf5_file = h5py.File(features_path,'r') 
    features = hdf5_file['features'][:]
    coords = hdf5_file['coords'][:]
    for idx in enumerate(features):
        with torch.no_grad():
            np_feature = features[idx]
            feature = torch.from_numpy(np_feature)
            if attn_save_path is not None:
                if tile_prob:
                    all_c1_probs = model(feature, tile_prob=True)
                    asset_dict = {'tile_probabilities': all_c1_probs, 'coords': coords}
                    mode = "a"
                    save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
                else:
                    A = model(feature, attention_only=True)

                    if A.size(0) > 1: #CLAM multi-branch attention
                        A = A[clam_pred]

                    A = A.view(-1, 1).cpu().numpy()

                    if ref_scores is not None:
                        for score_idx in range(len(A)):
                            A[score_idx] = score2percentile(A[score_idx], ref_scores)
                            
                    save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
                    asset_dict = {'attention_scores': A, 'coords': coords}
                    mode = "a"

    return save_path, wsi_object

def compute_from_patches(wsi_object, clam_pred=None, model=None, feature_extractor=None, batch_size=512,  
    attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
    top_left = wsi_kwargs['top_left']
    bot_right = wsi_kwargs['bot_right']
    patch_size = wsi_kwargs['patch_size']
    
    roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=8)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', len(roi_loader))
    mode = "w"

    for idx, (roi, coords) in enumerate(roi_loader):
        roi = roi.to(device)
        coords = coords.numpy()
        
        with torch.no_grad():
            features = feature_extractor(roi)

            if attn_save_path is not None:
                A = model(features, attention_only=True)
           
                if A.size(0) > 1: #CLAM multi-branch attention
                    A = A[clam_pred]

                A = A.view(-1, 1).cpu().numpy()

                if ref_scores is not None:
                    for score_idx in range(len(A)):
                        A[score_idx] = score2percentile(A[score_idx], ref_scores)

                asset_dict = {'attention_scores': A, 'coords': coords}
                save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
    
        if idx % math.ceil(num_batches * 0.05) == 0:
            print('processed {} / {}'.format(idx, num_batches))

        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        mode = "a"
    return attn_save_path, feat_save_path, wsi_object
