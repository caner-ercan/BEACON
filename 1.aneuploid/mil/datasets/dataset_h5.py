from __future__ import print_function, division
import sys
#sys.path.insert(0, '/rsrch5/home/trans_mol_path/cercan/.conda/envs/clam_albm/lib/python3.7/site-packages/')
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py
import albumentations as A

from random import randrange

# def eval_transforms(pretrained=False):
# 	if pretrained:
# 		mean = (0.485, 0.456, 0.406)
# 		std = (0.229, 0.224, 0.225)

# 	else:
# 		mean = (0.5,0.5,0.5)
# 		std = (0.5,0.5,0.5)

# 	trnsfrms_val = transforms.Compose(
# 					[
# 					 transforms.ToTensor(),
# 					 transforms.Normalize(mean = mean, std = std)
# 					]
# 				)

# 	return trnsfrms_val


# def aug_transforms():
# 	mean = (0.485, 0.456, 0.406)
# 	std = (0.229, 0.224, 0.225)
# 	trnsfrms_val = transforms.Compose(
# 				[
# 					transforms.RandomChoice([
# 						transforms.ColorJitter(brightness=[0.8,1.2]), # type: ignore
# 						transforms.ColorJitter(saturation=[0.8,1.2]),
# 						transforms.ColorJitter(brightness=[0.8,1.2], contrast = [0.8,1.2])]),
# 					transforms.RandomHorizontalFlip(p=0.8),
# 					transforms.ToTensor(),
# 					transforms.Normalize(mean = mean, std = std)
# 				]
# 			)
# 	return trnsfrms_val

# def alb_transforms():
# 	mean = (0.485, 0.456, 0.406)
# 	std = (0.229, 0.224, 0.225)

# 	trnsfrms_val = A.Compose([
# 		A.HorizontalFlip(p=0.5),
# 		A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
# 		A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
# 		A.MedianBlur(blur_limit=3, always_apply=False, p=0.5), # type: ignore
# 		A.ShiftScaleRotate(shift_limit=0, 
#                        scale_limit=0.1, 
#                        rotate_limit=0, 
#                        p=0.1),
# 		A.RandomRotate90(p=0.3),
# 		A.HueSaturationValue(p=0.2),
# 		A.RandomGamma (gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),
# 		A.Normalize(mean=mean, std=std)
# 	])

# 	return trnsfrms_val




class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		pretrained=False,
		custom_transforms=None,
		target_patch_size=-1,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.pretrained=pretrained
		if target_patch_size > 0:
			self.target_patch_size = (target_patch_size, target_patch_size)
		else:
			self.target_patch_size = None

		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['imgs']
		for name, value in dset.attrs.items():
			print(name, value)

		print('pretrained:', self.pretrained)
		print('transformations:', self.roi_transforms)
		if self.target_patch_size is not None:
			print('target_size: ', self.target_patch_size)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		pretrained=False,
		custom_transforms="normalisation",
		custom_downsample=1,
		target_patch_size=-1,
		backend="torch",
		normalisation="none"
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
		self.pretrained=pretrained
		self.wsi = wsi
		self.custom_transforms = custom_transforms
		self.backend = backend
		self.normalisation = normalisation


		#self.custom_transforms = "aug_transforms"
		#custom_transforms == "aug_transforms"
		#if custom_transforms is None:
		# self.roi_transforms = eval_transforms(pretrained=pretrained)
		# print("normalisation is applied")
		# elif custom_transforms == "aug_transforms":
		#self.roi_transforms = aug_transforms()
		#print("pytorch transformations were applied")
		# elif custom_transforms == "alb_transforms":
		#print("alb_transforms were applied")
		#self.roi_transforms = alb_transforms()
		# else:
		# 	self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size, ) * 2
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			else:
				self.target_patch_size = None
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('target patch size: ', self.target_patch_size)
		print('pretrained: ', self.pretrained)
		#print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		
		np_image = np.array(img)

		transforms = A.Compose([])

		if not self.custom_transforms == "none":

			if self.custom_transforms == "alb_transforms":
				#img.shape = (self.target_patch_size, self.target_patch_size,3)
				#img.dtype = 'np.uint8'
				#img.ndim=3
				if idx ==0:
					print("heavy transformations were applied")

				#np_image
				# np_image = np.array(img)
				# mean = (0.485, 0.456, 0.406)
				# std = (0.229, 0.224, 0.225)

				trnsfrms_aug = A.Compose([
					A.HorizontalFlip(p=0.5),
					A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
					A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
					A.GaussianBlur(blur_limit=(3,7), always_apply=False, p=0.5),
					A.ShiftScaleRotate(shift_limit=0, 
								scale_limit=0.1, 
								rotate_limit=0, 
								p=0.1),
					A.RandomRotate90(p=0.3),
					A.HueSaturationValue(p=0.2),
					A.RandomGamma (gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),
					# A.Normalize(mean=mean, std=std) 
				])

				transforms = A.Compose([
					*trnsfrms_aug.transforms
					])

			if self.normalisation == "imageNet":
				if idx ==0:
					print("imagenet color normalisation was applied")
				# color normalisation
				# if self.backend == "torch":
				mean = (0.485, 0.456, 0.406)
				std = (0.229, 0.224, 0.225)

				normalisation = A.Compose([
					A.Normalize(mean=mean, std=std)
				])
				transforms = A.Compose([
					*transforms.transforms,
					*normalisation.transforms
				])

			img = transforms(image=np_image)['image']

			#transformed image in two steps
			# transformed_data = trnsfrms_val(image=np_image)
			# #img_data = self.roi_transforms(image = img)
			# img = transformed_data["image"]

			#back to nparray
			img = np.uint8(img)

		if self.backend == "torch":
			#to tensor
			img = torch.from_numpy(img).unsqueeze(0)
			#re order float32
			img = img.permute(0, 3, 1, 2).to(torch.float32)

			#img = Image.fromarray(transformed_image)
		elif self.backend == "tf":
			import tensorflow as tf
			# Convert NumPy array to TensorFlow tensor
			img = tf.convert_to_tensor(img, dtype=tf.uint8)

			# Add a batch dimension by expanding the dimensions
			img = tf.expand_dims(img, axis=0)

			#img = tf.transpose(img, perm=[0, 3, 1, 2])

			img = tf.cast(img, dtype=tf.float32)		

			#img = self.roi_transforms(image=img)
		# elif self.custom_transforms == "aug_transforms":
		# 	if idx ==0:
		# 		print("pytorch transformations were applied")
		# 	self.roi_transforms = aug_transforms()
		# 	img = self.roi_transforms(img).unsqueeze(0)
		# elif self.custom_transforms == "normalisation":
		# 	if idx ==0:
		# 		print("only normalisations were applied")
			
		# 	if self.backend == "torch":
		# 		self.roi_transforms = eval_transforms(pretrained=self.pretrained)
		# 		img = self.roi_transforms(img).unsqueeze(0)
		# 	elif self.backend == "tf":
		# 		import tensorflow as tf

		# 		np_image = np.array(img)
		# 		mean = (0.485, 0.456, 0.406)
		# 		std = (0.229, 0.224, 0.225)

		# 		trnsfrms_val = A.Compose([
		# 			A.Normalize(mean=mean, std=std)
		# 		])
		# 		img = tf.convert_to_tensor(img, dtype=tf.uint8)
		# 		img = tf.expand_dims(img, axis=0)
		# 		#img = tf.transpose(img, perm=[0, 3, 1, 2])
		# 		img = tf.cast(img, dtype=tf.float32)	

		return img, coord

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




