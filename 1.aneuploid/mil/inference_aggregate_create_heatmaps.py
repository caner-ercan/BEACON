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
from utils.eval_utils import initiate_model as initiate_model
from models.model_clam import CLAM_MB, CLAM_SB
from models.resnet_custom import resnet50_baseline
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches, compute_from_features
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5
import csv
import tensorflow as tf

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
					help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--process_list', type=str, default=None)
parser.add_argument('--feature_dir', type=str, default=None)
parser.add_argument('--use_extracted_features',  action='store_true', default=False)
parser.add_argument('--tile_prob',  action='store_true', default=False)
parser.add_argument('--start_sample',  type=int, default=False)
parser.add_argument('--sample_count',  type=int, default=None)
args = parser.parse_args()

print("tile_prob:")
print(args.tile_prob)


def read_settings(ckpt_path):
    # Get the directory of the input file
    directory = os.path.dirname(ckpt_path)

    # Construct the path to the settings.csv file in the same directory
    settings_path = os.path.join(directory, 'settings.csv')

    # Initialize variables to store values
    model_type = None
    model_size = None

    # Check if the settings file exists
    if os.path.exists(settings_path):
        # Read settings.csv file
        with open(settings_path, 'r') as csvfile:
            # Create a CSV reader
            reader = csv.DictReader(csvfile)

            # Iterate over rows and extract values based on column names
            for row in reader:
                args.model_type = row.get('model_type')
                args.model_size = row.get('model_size')

read_settings(args.ckpt_path)
print(args.model_type)
print(args.model_size)

def infer_single_slide(model, features, label, reverse_label_dict, k=1,tile_prob= False):
	# if tf.is_tensor(features):
	# 	numpy_array = features.numpy()
	# 	features = torch.tensor(numpy_array)
	features = features.to(device)
	with torch.no_grad():
		if isinstance(model, (CLAM_SB, CLAM_MB)):
			model_results_dict = model(features)
			logits, Y_prob, Y_hat, A, results_dict = model(features, tile_prob=tile_prob)


			Y_hat = Y_hat.item()

			if isinstance(model, (CLAM_MB,)):
				A = A[Y_hat]

			A = A.view(-1, 1).cpu().numpy()

		else:
			raise NotImplementedError

		print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))	
		
		probs, ids = torch.topk(Y_prob, k)
		probs = probs[-1].cpu().numpy()
		ids = ids[-1].cpu().numpy()
		preds_str = np.array([reverse_label_dict[idx] for idx in ids])

	return ids, preds_str, probs, A, results_dict

def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key] 
			val = dtype(val)
			if isinstance(val, str):
				if len(val) > 0:
					params[key] = val
			elif not np.isnan(val):
				params[key] = val
			else:
				pdb.set_trace()

	return params

def parse_config_dict(args, config_dict):
	if args.save_exp_code is not None:
		config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
	if args.overlap is not None:
		config_dict['patching_arguments']['overlap'] = args.overlap
	config_dict['args']['ckpt_path']=args.ckpt_path
	config_dict['args']['data_dir']=args.data_dir
	config_dict['args']['process_list']=args.process_list
	config_dict['args']['save_dir']=args.save_dir
	config_dict['args']['feature_dir']=args.feature_dir
	config_dict['args']['tile_prob']=args.tile_prob
	config_dict['args']['sample_count']=args.sample_count
	config_dict['args']['start_sample']=args.start_sample
	config_dict['args']['use_extracted_features']=args.use_extracted_features
	config_dict['model_arguments']['model_size']=args.model_size
	config_dict['model_arguments']['model_type']=args.model_type
	config_dict['model_arguments']['data_root_dir']=args.feature_dir

	


	return config_dict

def collect_probabilities_df(asset_dict, patch_size=(224,224), stride_factor=0.5):
	"""
	Collects and averages probabilities and attention scores from sliding windows across subregions
	using a pandas DataFrame for more efficient manipulation.

	sliding_windows: List of (top_left_coords, probability, attention_score) for each window
	patch_size: Size of each patch (tuple: (height, width))
	stride_factor: How much each window moves, 0.5 means half patch overlap

	Returns:
	- avg_df: DataFrame with averaged probabilities and attention scores for each subregion
	"""    
	stride_h = int(patch_size[0] * stride_factor)  # Vertical stride
	stride_w = int(patch_size[1] * stride_factor)  # Horizontal stride

	coords = asset_dict['coords']
	attention_scores =asset_dict['attention_scores']
	attention_scores = attention_scores.flatten()
	# weighted_tile_probabilities = asset_dict['weighted_tile_probabilities']
	# tile_probabilities = asset_dict['tile_probabilities']
	raw_attention_scores = asset_dict['raw_attention_scores']
	raw_attention_scores = raw_attention_scores.flatten()
	attention_scores_minmax = asset_dict['attention_scores_minmax']
	attention_scores_minmax = attention_scores_minmax.flatten()
	A_raw_minmax = asset_dict['A_raw_minmax']
	A_raw_minmax = A_raw_minmax.flatten()

	


	# Prepare a list to collect data for the DataFrame
	data = []

	# for (top_left, prob, attn_score) in h5data:
	for idx in range(len(coords)):
		x, y = coords[idx]
		
		# Define 4 subregions (assuming 1/4 division of patch)
		subregions = [
			(x, y),  # Top-left
			(x + stride_w, y),  # Top-right
			(x, y + stride_h),  # Bottom-left
			(x + stride_w, y + stride_h),  # Bottom-right
		]
		
		for region in subregions:
			data.append({
				'coords': region,
				# 'weighted_tile_probabilities': weighted_tile_probabilities[idx],
				# 'tile_probabilities': tile_probabilities[idx],
				'attention_scores': attention_scores[idx],
				'raw_attention_scores': raw_attention_scores[idx],
				# 'attention_scores_minmax': attention_scores_minmax[idx],
				'A_raw_minmax': A_raw_minmax[idx]
			})

	# Convert data to a pandas DataFrame
	df = pd.DataFrame(data)
	# Group by 'region' and compute mean for each region
	# avg_df = df.groupby('coords').mean().reset_index()

	# print(df['tile_probabilities'].head())
	# # print(df['attention_scores'].head())
	# print(df['A_raw_minmax'].head())
	# print(df['attention_scores_minmax'].head())

	avg_df = df.groupby('coords').agg(
		# avg_probability=('tile_probabilities', 'mean'),
		avg_attention_score=('attention_scores', 'mean'),
		# avg_weighted_tile_prob=('weighted_tile_probabilities', 'mean'),
		avg_raw_attention_score=('raw_attention_scores', 'mean'),
		# avg_attention_scores_minmax=('attention_scores_minmax', 'mean'),
		avg_A_raw_minmax=('A_raw_minmax', 'mean')
	).reset_index()

	attention_scores_minmax = min_max_normalization(avg_df['avg_attention_score'])
	avg_df['avg_attention_scores_minmax'] = attention_scores_minmax

	avg_asset_dict = { 'coords': np.array(avg_df['coords'].tolist(), dtype=int), 
			# 'avg_tile_probability': np.array(avg_df['avg_probability'].tolist(), dtype=float), 
			'avg_attention_score': np.array(avg_df['avg_attention_score'].tolist(), dtype=float), 
			# 'avg_weighted_tile_prob':np.array(avg_df['avg_weighted_tile_prob'].tolist(), dtype=float), 
			'avg_raw_attention_score':np.array(avg_df['avg_raw_attention_score'].tolist(), dtype=float), 
			'avg_attention_scores_minmax':np.array(avg_df['avg_attention_scores_minmax'].tolist(), dtype=float), 
			'avg_A_raw_minmax':np.array(avg_df['avg_A_raw_minmax'].tolist(), dtype=float), 
			'attention_scores': attention_scores 
			# 'tile_probabilities':tile_probabilities,
			# 'weighted_tile_probabilities':weighted_tile_probabilities
			}
		

	return avg_asset_dict

def min_max_normalization(values):
    
    min_value = np.min(values)
    max_value = np.max(values)
    
    normalized_values = (values - min_value) / (max_value - min_value)
    
    return normalized_values

if __name__ == '__main__':
	#config_path = os.path.join('heatmaps/configs', args.config_file)
	config_dict = yaml.safe_load(open(args.config_file, 'r'))
	config_dict = parse_config_dict(args, config_dict)

	for key, value in config_dict.items():
		if isinstance(value, dict):
			print('\n'+key)
			for value_key, value_value in value.items():
				print (value_key + " : " + str(value_value))
		else:
			print ('\n'+key + " : " + str(value))	
#	decision = input('Continue? Y/N ')
#	if decision in ['Y', 'y', 'Yes', 'yes']:
#		pass
#	elif decision in ['N', 'n', 'No', 'NO']:
#		exit()
#	else:
#		raise NotImplementedError
	args = config_dict
	patch_args = argparse.Namespace(**args['patching_arguments'])
	data_args = argparse.Namespace(**args['data_arguments'])
	model_args = args['model_arguments']
	model_args.update({'n_classes': args['exp_arguments']['n_classes']})
	model_args = argparse.Namespace(**model_args)
	exp_args = argparse.Namespace(**args['exp_arguments'])
	heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
	sample_args = argparse.Namespace(**args['sample_arguments'])
	args = argparse.Namespace(**args['args'])

	patch_size = tuple([patch_args.patch_size for i in range(2)])
	step_size = tuple((np.array(patch_size) * (1-patch_args.overlap)).astype(int))
	print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))

	
	preset = data_args.preset
	def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
					  'keep_ids': 'none', 'exclude_ids':'none', 'gaus_otsu': False}
	def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
	def_vis_params = {'vis_level': -1, 'line_thickness': 250}
	def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if preset is not None:
		preset_df = pd.read_csv(preset)
		for key in def_seg_params.keys():
			def_seg_params[key] = preset_df.loc[0, key]

		for key in def_filter_params.keys():
			def_filter_params[key] = preset_df.loc[0, key]

		for key in def_vis_params.keys():
			def_vis_params[key] = preset_df.loc[0, key]

		for key in def_patch_params.keys():
			def_patch_params[key] = preset_df.loc[0, key]

	
	if args.process_list is None:
		if isinstance(args.data_dir, list):
			slides = []
			for data_dir in args.data_dir:
				slides.extend(os.listdir(data_dir))
		else:
			slides = sorted(os.listdir(args.data_dir))
		slides = [slide for slide in slides if data_args.slide_ext in slide]
		df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
		
	else:
		#df = pd.read_csv(os.path.join('heatmaps/process_lists', args.process_list))
		df = pd.read_csv(args.process_list)
		df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

	mask = df['process'] == 1
	process_stack = df[mask].reset_index(drop=True)
	total = len(process_stack)
	# print('\nlist of slides to process: ')
	# print(process_stack.head(len(process_stack)))

	print('\ninitializing model from checkpoint')
	#ckpt_path = model_args.ckpt_path
	ckpt_path = args.ckpt_path
	print('\nckpt path: {}'.format(ckpt_path))
	# model_args.data_root_dir = args.feature_dir
	if model_args.initiate_fn == 'initiate_model':
		print("model arguments: ")
		print(model_args)
		model =  initiate_model(model_args, ckpt_path)
	else:
		raise NotImplementedError

	if not args.use_extracted_features:
		feature_extractor = resnet50_baseline(pretrained=True)
		feature_extractor.eval()
		device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print('Done!')
		if torch.cuda.device_count() > 1:
			device_ids = list(range(torch.cuda.device_count()))
			feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids).to('cuda:0')
		else:
			feature_extractor = feature_extractor.to(device)
	else:
		feature_extractor = None


	label_dict =  data_args.label_dict
	class_labels = list(label_dict.keys())
	class_encodings = list(label_dict.values())
	reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 



	exp_args.production_save_dir=os.path.join(args.save_dir, "production")
	exp_args.raw_save_dir=os.path.join(args.save_dir, "raw")
	os.makedirs(exp_args.production_save_dir, exist_ok=True)
	os.makedirs(exp_args.raw_save_dir, exist_ok=True)
	blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
	'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}
	print("start sample: {}".format(args.start_sample))
	if args.start_sample is not None:

		process_stack = process_stack.iloc[args.start_sample:args.start_sample+args.sample_count].reset_index(drop=True)
		print(process_stack)

	for i in range(len(process_stack)):
		
		slide_name = process_stack.loc[i, 'slide_id']
		if data_args.slide_ext not in slide_name:
			slide_name+=data_args.slide_ext
		print('\nprocessing: ', slide_name)	
		print(i+1,'/', len(process_stack))	

		try:
			label = process_stack.loc[i, 'label']
		except KeyError:
			label = 'Unspecified'

		slide_id = slide_name.replace(data_args.slide_ext, '')

		if not isinstance(label, str):
			grouping = reverse_label_dict[label]
		else:
			grouping = label


		
		if args.use_extracted_features:
			features_path = os.path.join(args.feature_dir,'pt_files', slide_id+'.pt')
			h5_path = os.path.join(args.feature_dir,'h5_files', slide_id+'.h5')
		else:
			features_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
			h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')
		print(h5_path)

		p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
		os.makedirs(p_slide_save_dir, exist_ok=True)

		r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping),  slide_id)
		os.makedirs(r_slide_save_dir, exist_ok=True)

		block_map_save_path = os.path.join(r_slide_save_dir, '{}_{}px_blockmap.h5'.format(slide_id,patch_size[0]))

		if os.path.exists(block_map_save_path):
			print(f"img path exists! \nreading from h5file")
			with h5py.File(block_map_save_path, 'r') as file:
				asset_dict = {key: file[key][:] for key in file.keys()}
					
		else:
			with h5py.File(h5_path, "r") as file:
			
				features = torch.tensor(file['features'][:])
				coords = file['coords'][:]
				print(f"feature size: {str(features.shape)}")
				#torch.save(features, features_path)
				# file.close()
			process_stack.loc[i, 'bag_size'] = len(features)

			Y_hats, Y_hats_str, Y_probs, A, results_dict = infer_single_slide(model, features, label, reverse_label_dict, exp_args.n_classes, tile_prob=args.tile_prob)

			# asset_dict = {'raw_attention_scores':results_dict['A_raw'], 'coords': coords, "tile_probabilities":results_dict['h_probs'], "weighted_tile_probabilities":results_dict['h_weighted']}
			asset_dict = {'coords': coords}

			A_raw = results_dict['A_raw']
			# A_raw_minmax = min_max_normalization(A_raw)

			A_raw = torch.tensor(results_dict['A_raw']).float()
			A_softmax = F.softmax(A_raw, dim=0)
			A_softmax = np.array(A_softmax)
			A_softmax_minmax = min_max_normalization(A_softmax)

			# asset_dict['attention_scores'] = A_softmax
			asset_dict['attention_scores_minmax'] = A_softmax_minmax
			# asset_dict['A_raw_minmax'] = A_raw_minmax

			block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')

		

		## aggregate regions
		# avg_asset_dict = collect_probabilities_df(asset_dict)
		# aggregated_block_map_save_path = os.path.join(r_slide_save_dir, '{}_{}px_aggregated_blockmap.h5'.format(slide_id,patch_size[0])) # this patch size needs to be divided if sliding windows is used.
		# aggregated_block_map_save_path = save_hdf5(aggregated_block_map_save_path, avg_asset_dict, mode='w')


		# create heatmaps
		heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
		if not os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
			coord_dset = avg_asset_dict['coords'][:]
			avg_attention_scores = avg_asset_dict['avg_attention_score']
			# avg_weighted_tile_prob = avg_asset_dict['avg_weighted_tile_prob']
			# avg_tile_probability = avg_asset_dict['avg_tile_probability']
			avg_raw_attention_score = avg_asset_dict['avg_raw_attention_score']
			avg_attention_scores_minmax = avg_asset_dict['avg_attention_scores_minmax']
			avg_A_raw_minmax = avg_asset_dict['avg_A_raw_minmax']
			scores = avg_raw_attention_score

			if isinstance(args.data_dir, str):
				slide_path = os.path.join(args.data_dir, slide_name)
			elif isinstance(args.data_dir, dict):
				data_dir_key = process_stack.loc[i, data_args.data_dir_key]
				slide_path = os.path.join(args.data_dir[data_dir_key], slide_name)
			else:
				raise NotImplementedError

			mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
			
			# Load segmentation and filter parameters
			seg_params = def_seg_params.copy()
			filter_params = def_filter_params.copy()
			vis_params = def_vis_params.copy()

			seg_params = load_params(process_stack.loc[i], seg_params)
			filter_params = load_params(process_stack.loc[i], filter_params)
			vis_params = load_params(process_stack.loc[i], vis_params)

			keep_ids = str(seg_params['keep_ids'])
			if len(keep_ids) > 0 and keep_ids != 'none':
				seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
			else:
				seg_params['keep_ids'] = []

			exclude_ids = str(seg_params['exclude_ids'])
			if len(exclude_ids) > 0 and exclude_ids != 'none':
				seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
			else:
				seg_params['exclude_ids'] = []
			if i == 0:
				for key, val in seg_params.items():
					print('{}: {}'.format(key, val))

				for key, val in filter_params.items():
					print('{}: {}'.format(key, val))

				for key, val in vis_params.items():
					print('{}: {}'.format(key, val))
			if heatmap_args.use_roi:
				x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
				y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
				top_left = (int(x1), int(y1))
				bot_right = (int(x2), int(y2))
			else:
				top_left = None
				bot_right = None

			print('Initializing WSI object')
			wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
			print('Done!')

			wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

			# the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
			vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

		
			mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
			if vis_params['vis_level'] < 0:
				best_level = wsi_object.wsi.get_best_level_for_downsample(64)
				vis_params['vis_level'] = best_level
			mask = wsi_object.visWSI(**vis_params, number_contours=True)
			mask.save(mask_path)



			wsi_object.saveSegmentation(mask_file)

			# save top 3 predictions
			for c in range(exp_args.n_classes):
				process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
				process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]



			samples = sample_args.samples
			for sample in samples:
				if sample['sample']:
					tag = "label_{}_pred_{}".format(label, Y_hats[0])
					sample_save_dir =  os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', str(tag), sample['name'])
					os.makedirs(sample_save_dir, exist_ok=True)
					print('sampling {}'.format(sample['name']))
					sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
						score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
					for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
						print('coord: {} score: {:.3f}'.format(s_coord, s_score))
						patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
						patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.tif'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))

			wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
			'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

			# Attention heatmap
			heatmap = drawHeatmap(scores, coord_dset, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
							thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True, map_any =False)
		
			heatmap.save(os.path.join(r_slide_save_dir, '{}_{}_{}px_blockmap.png'.format('A_raw',slide_id,patch_size[0])))

			# h heatmap
			# print("h_dset shape")
			# print(h_dset.shape)
			# heatmap = drawHeatmap(avg_tile_probability, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
			# 				thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=False)
		
			# heatmap.save(os.path.join(r_slide_save_dir, '{}_{}_{}px_blockmap.png'.format('h',slide_id,patch_size[0])))

			#h weighted heatmap
			# print("hw_dset shape")
			# print(hw_dset.shape)
			# heatmap = drawHeatmap(avg_weighted_tile_prob, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
			# 				thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=False)
		
			# heatmap.save(os.path.join(r_slide_save_dir, '{}_{}_{}px_blockmap.png'.format('h_weighted',slide_id,patch_size[0])))

			heatmap = drawHeatmap(avg_attention_scores, coord_dset, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
							thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=False)
		
			heatmap.save(os.path.join(r_slide_save_dir, '{}_{}_{}px_blockmap.png'.format('A',slide_id,patch_size[0])))



			heatmap = drawHeatmap(avg_A_raw_minmax, coord_dset, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
							thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=False)
		
			heatmap.save(os.path.join(r_slide_save_dir, '{}_{}_{}px_blockmap.png'.format('A_raw_minmax',slide_id,patch_size[0])))



			heatmap = drawHeatmap(avg_attention_scores_minmax, coord_dset, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
							thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=False)
		
			heatmap.save(os.path.join(r_slide_save_dir, '{}_{}_{}px_blockmap.png'.format('A_minmax',slide_id,patch_size[0])))

			del heatmap
		
		else:
			pass

		save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))

		if heatmap_args.use_ref_scores:
			ref_scores = scores
		else:
			ref_scores = None
		
		if heatmap_args.calc_heatmap:
			if not args.use_extracted_features:
				compute_from_patches(wsi_object=wsi_object, clam_pred=Y_hats[0], model=model, feature_extractor=feature_extractor, batch_size=exp_args.batch_size, **wsi_kwargs, 
								attn_save_path=save_path,  ref_scores=ref_scores,tile_prob=args.tile_prob)
			else:
				compute_from_features(wsi_object=wsi_object, clam_pred=Y_hats[0], model=model, features_path = h5_path,batch_size=exp_args.batch_size, **wsi_kwargs, 
								attn_save_path=save_path,  ref_scores=ref_scores,tile_prob=args.tile_prob) 

		if not os.path.isfile(save_path):
			print('heatmap {} not found'.format(save_path))
			if heatmap_args.use_roi:
				save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
				print('found heatmap for whole slide')
				save_path = save_path_full
			else:
				continue

		file = h5py.File(save_path, 'r')
		dset = file['attention_scores']
		coord_dset = file['coords']
		scores = dset[:]
		coords = coord_dset[:]
		file.close()

		heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}
		if heatmap_args.use_ref_scores:
			heatmap_vis_args['convert_to_percentiles'] = False

		heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(patch_args.overlap), int(heatmap_args.use_roi),
																						int(heatmap_args.blur), 
																						int(heatmap_args.use_ref_scores), int(heatmap_args.blank_canvas), 
																						float(heatmap_args.alpha), int(heatmap_args.vis_level), 
																						int(heatmap_args.binarize), float(heatmap_args.binary_thresh), heatmap_args.save_ext)


		if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
			pass
		
		else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
			heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,  
								cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args, 
								binarize=heatmap_args.binarize, 
								blank_canvas=heatmap_args.blank_canvas,
								thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size,
								overlap=patch_args.overlap, 
								top_left=top_left, bot_right = bot_right)
			if heatmap_args.save_ext == 'jpg':
				heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
			else:
				heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
		
		if heatmap_args.save_orig:
			if heatmap_args.vis_level >= 0:
				vis_level = heatmap_args.vis_level
			else:
				vis_level = vis_params['vis_level']
			heatmap_save_name = '{}_orig_{}.{}'.format(slide_id,int(vis_level), heatmap_args.save_ext)
			if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
				pass
			else:
				heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
				if heatmap_args.save_ext == 'jpg':
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
				else:
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))

with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
	yaml.dump(config_dict, outfile, default_flow_style=False)


			








			
			
			# print('slide id: ', slide_id)
			# print('top left: ', top_left, ' bot right: ', bot_right)

			
			

		

			# ##### check if h5_features_file exists ######
			# if not os.path.isfile(h5_path) :
			# 	_, _, wsi_object = compute_from_patches(wsi_object=wsi_object, 
			# 									model=model, 
			# 									feature_extractor=feature_extractor, 
			# 									batch_size=exp_args.batch_size, **blocky_wsi_kwargs, 
			# 									attn_save_path=None, feat_save_path=h5_path, 
			# 									ref_scores=None)		
			
			
			# # load features 
			# ##### check if pt_features_file exists ######
			# # if not os.path.isfile(features_path):
			# if True:
			# 	file = h5py.File(h5_path, "r")
			# 	features = torch.tensor(file['features'][:])
			# 	print(f"feature size: {str(features.shape)}")
			# 	#torch.save(features, features_path)
			# 	file.close()
			# else:
			# 	features = torch.load(features_path)

			
			
			
				
			
				# print("##################"), print(f"feature size: {str(features.size)}"),print("##################") 
				# print("resul/ts_dict")
				# print(results_dict["probs"])
				# print("A")
				# print(A.dtype)
				# print("h")
				# print(results_dict['h_probs'].dtype)
				# print("M")
				# print(results_dict["M"].shape)
				# print("h_weighted")
				# print(results_dict["h_weighted"].dtype)
				# print("##################"), print(f"score size: {str(A.size)}"),print("##################") 

			# del features
			# if not os.path.isfile(block_map_save_path): 
			# 	file = h5py.File(h5_path, "r")
				
			# 	file.close()
			# 					# print(results_dict['h_probs'])
			# 	# print("##################"), print(f"score size: {str(asset_dict['attention_scores'].shape)}"),print("##################") 
				
			

		# os.makedirs('heatmaps/results/', exist_ok=True)
if args.process_list is not None:
	#process_stack.to_csv('heatmaps/results/{}.csv'.format(args.process_list.replace('.csv', '')), index=False)
	process_stack.to_csv(os.path.join(exp_args.production_save_dir,os.path.basename(args.process_list)), index=False)
else:
	process_stack.to_csv('heatmaps/results/{}.csv'.format(exp_args.save_exp_code), index=False)
	
			

