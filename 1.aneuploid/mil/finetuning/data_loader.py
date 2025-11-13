import os
import glob
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import parse_image_mask, get_crops_bounds, generate_random_crop_boundary
from augmenter import simsiam_augmenter
import albumentations as A
import numpy as np


def get_train_val_lists(csv_path):
    slide_list = pd.read_csv(csv_path)
    slide_list = slide_list.drop_duplicates(subset='slide_id', keep='first')
    aneuploid_list = slide_list.loc[slide_list['label'] == 'diploid', 'slide_id']
    diploid_list = slide_list.loc[slide_list['label'] == 'diploid', 'slide_id']

    train_aneuploid, val_aneuploid = train_test_split(aneuploid_list, test_size=0.2, random_state=42)
    train_diploid, val_diploid = train_test_split(diploid_list, test_size=0.2, random_state=42)
    train_list = pd.concat([train_aneuploid, train_diploid], ignore_index=True)
    val_list = pd.concat([val_aneuploid, val_diploid], ignore_index=True)

    return train_list, val_list

def get_tif_files(base_path, csv_path):
    train_list, val_list = get_train_val_lists(csv_path)
    tif_files_train = []
    tif_files_val = []
    for slide_id in train_list:
        tif_files_train += glob.glob(f"{base_path}/{slide_id}*.tif")
    for slide_id in val_list:
        tif_files_val += glob.glob(f"{base_path}/{slide_id}*.tif")
    print(f"Number of training biopsies: {len(train_list)}")
    print(f"Number of validation biopsies: {len(val_list)}")

    return tif_files_train, tif_files_val

def create_random_square(width_dim=None,height_dim=None ):
    if width_dim is None:
        width_dim = height_dim
    return A.Compose([
        A.RandomCrop(width=width_dim, height=height_dim, p=1.0, pad_if_needed = True),
    ])
    
def check_tissue_content(mask, threshold =0.5):
    
    tissue_pixels = np.sum(mask > 0)
    total_pixels = mask.size

    return (tissue_pixels  / total_pixels) >= threshold


def get_crops(filename, region_dim=1024, crop_dim=448):
    wsi_img, wsi_mask = parse_image_mask(filename)
    valid_patch = False
    crops = []

    width_dim = min(wsi_img.shape[1], region_dim)
    height_dim = min(wsi_img.shape[0], region_dim)
        
    get_region = create_random_square(width_dim , height_dim)
    get_crop = create_random_square(crop_dim)

    threshold = 0.5
    max_attempts = 10

    while not valid_patch:
        region = get_region(image=wsi_img, mask=wsi_mask)
        region_mask = region['mask']
        region_image = region['image']
        print(filename)
        # print(region_image.shape)

        for _ in range(max_attempts):
            try:
                crop = get_crop(image=region_image, mask=region_mask)
            except RuntimeError as e:
                print(e)
                print('Error in crop '+ {filename})
            crop_test = check_tissue_content(crop['mask'], threshold)
            if crop_test:
                img_crop = tf.convert_to_tensor(crop['image'], dtype=tf.float32)
                crops.append(img_crop)
                if len(crops) == 2:
                    valid_patch = True
                    break
            else:
                threshold -= 0.1

    return crops


    # while not valid_patch:
    #     augmented = transform(image=wsi_img, mask=wsi_mask)
    #     img_crop = augmented['image']
    #     mask_crop = augmented['mask']

    #     tissue_pixels = np.sum(mask_crop > 0)
    #     total_pixels = mask_crop.size
    #     if (tissue_pixels / total_pixels) >= 0.5:
    #         img_crop = tf.convert_to_tensor(img_crop, dtype=tf.float32)
    #         crops.append(img_crop)

    # return crops

# def get_crops(filename, region_dim=1024, crop_dim=448):
#     wsi_img, wsi_mask = parse_image_mask(filename)
#     valid_patch = False
#     crops = []
#     while not valid_patch:
#         region_start_y, region_start_x, end_y, end_x = generate_random_crop_boundary(wsi_mask.shape, region_dim)
#         region_mask = wsi_mask[region_start_y:end_y, region_start_x:end_x].copy()
#         crop_bounds = get_crops_bounds(region_mask, region_dim=region_dim, crop_dim=crop_dim)
#         if crop_bounds is not None:
#             region_img = wsi_img[region_start_y:end_y, region_start_x:end_x].copy()
#             img_crop = region_img[crop_bounds[0]:crop_bounds[0] + crop_dim, crop_bounds[1]:crop_bounds[1] + crop_dim].copy()
#             img_crop = tf.convert_to_tensor(img_crop, dtype=tf.float32)
#             crops.append(img_crop)
#             if len(crops) == 2:
#                 valid_patch = True
#     return crops

def get_views(file_path):
    file_path = file_path.numpy().decode()
    crops = get_crops(file_path)
    view1 = simsiam_augmenter(crops[0])
    view2 = simsiam_augmenter(crops[1])
    view1.set_shape((224, 224, 3))
    view2.set_shape((224, 224, 3))
    return view1, view2

@tf.function
def tf_read_slide(file_path):
    view1, view2 = tf.py_function(get_views, [file_path], [tf.float32, tf.float32])
    view1.set_shape((224, 224, 3))
    view2.set_shape((224, 224, 3))
    return view1, view2

def create_datasets(tif_files_train, tif_files_val, batch_size=32):
    train_ds = tf.data.Dataset.from_tensor_slices(tif_files_train)
    train_ds = train_ds.repeat()
    train_ds = train_ds.shuffle(1024)
    train_ds = train_ds.map(tf_read_slide, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(tif_files_val)
    val_ds = val_ds.repeat()
    val_ds = val_ds.shuffle(1024)
    val_ds = val_ds.map(tf_read_slide, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds