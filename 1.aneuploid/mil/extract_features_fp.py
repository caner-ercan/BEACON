import sys
import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
import argparse
from utils.utils import print_network, collate_features, tf_collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_w_loader(file_path, feat_dir, bag_name, wsi, model, custom_transforms="normalisation",
 	batch_size = 1, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, custom_transforms=custom_transforms,
		custom_downsample=custom_downsample, target_patch_size=target_patch_size, backend="torch")
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			torch.save(features, os.path.join(feat_dir, 'pt_files', bag_base+'.pt'))

			features = features.cpu().numpy()
			# activate this for the augmented run
			#coords=coords*2

			asset_dict = {'features': features, 'coords': coords}
			print('features size: ', features.shape)
			print('coordinates size: ', coords.shape)
			save_hdf5(os.path.join(feat_dir, 'h5_files', bag_name), asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return h5_output_path

def compute_w_loader_tf(file_path, feat_dir, bag_name, wsi, model, custom_transforms="normalisation",
 	batch_size = 1, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1, normalisation=None):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	import tensorflow as tf
	
	print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
	if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
		tf.device('/GPU:0')
	
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, custom_transforms=custom_transforms,
		custom_downsample=custom_downsample, target_patch_size=target_patch_size, backend="tf", normalisation = normalisation)
	#x, y = dataset[0]
	#loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)
	# tf_dataset = tf.data.Dataset.from_generator(
	# 	lambda: iter(dataset),
	# 	output_types=(tf.float32, tf.int64),
	#     output_shapes=((None,1024), (None,2))
	# 	)
	#tf_dataset = tf.data.Dataset.zip(dataset[0])
	#img = tf.constant(dataset[0]) p
	# print(type(dataset))

	# for i in range(len(dataset)):
	# 	print(f"Element {i}:")
	# 	print(f"Type: {type(dataset[i])}")
	# 	print(f"len: {len(dataset[i])}") 
	# 	print(f"Tvalue: {dataset[i]}") 



	#for img, coord in dataset:
	# def dataset_generator(dataset):
	images = []
	coordinates = []
	for i in range(len(dataset)):
		images.append(dataset[i][0])
		coordinates.append(dataset[i][1])
		# return images, coordinates

	images = np.squeeze(images, axis=1)


	# tf_dataset = tf.data.Dataset.from_generator(
	# 	lambda: dataset_generator(dataset),
	# 	output_types=(tf.float32, tf.int64),
	#     output_shapes=((None,1024), (None,2))
	# 	)
	# tf_dataset = tf.data.Dataset.from_tensor_slices((images, coordinates ))
	# print(images)
	# print(coordinates)


	# images_dataset = tf_dataset.from_tensor_slices(images)
	images = tf.convert_to_tensor(images, np.float32)
	coordinates = tf.convert_to_tensor(coordinates, np.int64)
	# print(images)
	# print(coordinates)

	# coordinates_dataset = tf_dataset.from_tensor_slices(coordinates)

	# zipped_dataset = tf.data.Dataset.zip((images_dataset, coordinates_dataset))

	#if this one wont work, try the tf_collate_features
	# images_dataset,coordinates_dataset =  tf_collate_features(dataset)


	# images_dataset = tf.constant(images)
	# coordinates_dataset = tf.constant(coordinates)
	images.shape
	tf_dataset = tf.data.Dataset.from_tensor_slices((images, coordinates ))


	
	
	tf_dataset = tf_dataset.batch(batch_size)
	tf_dataset = tf_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
	#print('processing {}: total of {} batches'.format(file_path,len(tf_dataset)))
	mode = 'w'
	bag_base, _ = os.path.splitext(bag_name)

	for count, (batch, coords) in enumerate(tf_dataset):
		# if count % print_every == 0:
		print('batch {}/{}, {} files processed'.format(count, len(tf_dataset), count * batch_size))
			#print(batch)
		with tf.GradientTape() as tape:
			features = model(batch)
			torch.save(features, os.path.join(feat_dir, 'pt_files', bag_base+'.pt'))
			features = features.cpu().numpy()
			coords = coords.numpy()

			asset_dict = {'features': features, 'coords': coords}
			print('features size: ', features.shape)
			print('coordinates size: ', coords.shape)
			save_hdf5(os.path.join(feat_dir, 'h5_files', bag_name), asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)

	return output_path



parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--transforms', choices=["normalisation", 'aug_transforms', 'alb_transforms', 'none'], default="normalisation")
parser.add_argument('--arch',type=str, default="resnet50", choices=['resnet50', 'densenet121', "path-50x1-remedis-m","path-50x1-remedis-s", "path-152x2-remedis-m", "path-152x2-remedis-s"])
parser.add_argument('--seed',type=int, default=0)
parser.add_argument('--first_idx',type=int, default=0)
parser.add_argument('--no_linear',default=False, action='store_true')
parser.add_argument('--noAverage',default=False, action='store_true')
parser.add_argument('--normalisation', choices=["none", 'imageNet', '05', 'TCGA'], default="none")
args = parser.parse_args()


if __name__ == '__main__':
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)   
	if args.seed == 0:
		args.transforms = "normalisation"

	settings = {'seed': args.seed,
			'data_h5_dir': args.data_h5_dir,
            'data_slide_dir': args.data_slide_dir, 
            'slide_ext': args.slide_ext,
            'csv_path': args.csv_path,
            'feat_dir': args.feat_dir,
            'batch_size': args.batch_size,
            'custom_downsample': args.custom_downsample,
            'target_patch_size': args.target_patch_size,
            'transforms': args.transforms,
            'arch': args.arch,
			'normalisation': args.normalisation}


	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	with open(args.feat_dir + '/settings.txt', 'w') as f:
		print(settings, file=f)
	f.close()

	print('loading model checkpoint')
	if args.arch in ["resnet50", "densenet50"]:
		from models.resnet_custom import resnet50_baseline
		import tensorflow as tf
		model = resnet50_baseline(arch = args.arch, pretrained=True)
		model = model.to(device)
		print_network(model)
		if torch.cuda.device_count() > 1:
			model = nn.DataParallel(model)
			
		model.eval()

	elif args.arch in ["path-50x1-remedis-m","path-50x1-remedis-s", "path-152x2-remedis-m", "path-152x2-remedis-s"] :
		from models.model_foundation import foundationModel
		import tensorflow as tf
		import keras
		keras.utils.set_random_seed(args.seed)
		tf.config.experimental.enable_op_determinism()


		if len(tf.config.list_physical_devices('GPU'))>0:
			device = "/gpu:0"
			print("GPU available. Using GPU.")
		else:
			device = "/cpu:0"
			print("GPU not available. Using CPU.")

		with tf.device(device):
			model = foundationModel(arch = args.arch, no_linear = args.no_linear, noAverage = args.noAverage)
			model.summary()
	else:
		raise NotImplementedError
	

	total = len(bags_dataset)

	# for bag_candidate_idx in range(total):
	for bag_candidate_idx in range(args.first_idx, args.first_idx + 20):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		#output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		if args.arch in ["resnet50", "densenet121"]:
			compute_w_loader(file_path=h5_file_path, feat_dir=args.feat_dir, bag_name=bag_name, wsi=wsi, 
			model = model, custom_transforms=args.transforms, batch_size = args.batch_size, verbose = 1, print_every = 20, 
			custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size, normalisation=args.normalisation)
		else:
			compute_w_loader_tf(file_path=h5_file_path, feat_dir=args.feat_dir, bag_name=bag_name, wsi=wsi, 
			model = model, custom_transforms=args.transforms, batch_size = args.batch_size, verbose = 1, print_every = 20, 
			custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size, normalisation=args.normalisation)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(bag_name, time_elapsed))
		# file = h5py.File(output_file_path, "r")

		# features = file['features'][:]
		# print('features size: ', features.shape)
		# print('coordinates size: ', file['coords'].shape)
		# features = torch.from_numpy(features)
		# bag_base, _ = os.path.splitext(bag_name)
		# torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))



