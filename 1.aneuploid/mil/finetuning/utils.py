import os
import random
import numpy as np
import cv2
import openslide
import tensorflow as tf

def parse_image_mask(file_path, sthresh=30):
    wsi = openslide.OpenSlide(file_path)
    level = wsi.get_best_level_for_downsample(64)
    image_ds = np.array(wsi.read_region((0, 0), level, wsi.level_dimensions[level]).convert('RGB'))
    img_gray = cv2.cvtColor(image_ds, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
    _, img_otsu = cv2.threshold(img_blur, sthresh, 220, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

    original_dimensions = wsi.level_dimensions[0]
    mask = cv2.resize(mask, (original_dimensions[0], original_dimensions[1]), interpolation=cv2.INTER_NEAREST)
    image = np.array(wsi.read_region((0, 0), 0, wsi.level_dimensions[0]).convert('RGB'))

    return image, mask

def generate_random_crop_boundary(image_shape, crop_size):
    height, width = image_shape[:2]
    if height < crop_size:
        start_y = 0
        end_y = height
    else:
        max_y = height - crop_size
        start_y = random.randint(0, max_y)
        end_y = start_y + crop_size

    if width < crop_size:
        start_x = 0
        end_x = width
    else:
        max_x = width - crop_size
        start_x = random.randint(0, max_x)
        end_x = start_x + crop_size

    return start_y, start_x, end_y, end_x

def check_tissue_content(mask, crop_start_y, crop_start_x, crop_dim=448, threshold=0.5):
    crop_mask = mask[crop_start_y:crop_start_y + crop_dim, crop_start_x:crop_start_x + crop_dim]
    tissue_pixels = np.sum(crop_mask > 0)
    total_pixels = crop_mask.size
    return (tissue_pixels / total_pixels) >= threshold

def get_crops_bounds(mask, region_dim=1024, crop_dim=448, max_attempts=20):
    threshold = 0.5
    for _ in range(max_attempts):
        crop_start_y, crop_start_x, _, _ = generate_random_crop_boundary((region_dim, region_dim), crop_dim)
        if check_tissue_content(mask, crop_start_y, crop_start_x):
            crop_bounds = (crop_start_y, crop_start_x)
            return crop_bounds
        else:
            threshold -= 0.05
    return None