import tensorflow as tf
import numpy as np
import cv2
import random
import openslide
import tensorflow_similarity as tfsim


def parse_image4mask(filename, sthresh =30):
    wsi = openslide.OpenSlide(filename)
    level = wsi.get_best_level_for_downsample(64)
    image = wsi.read_region((0, 0), level, wsi.level_dimensions[level])
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(7,7),0)
    _, img_otsu = cv2.threshold(img_blur, sthresh, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel) 

    original_dimensions = wsi.level_dimensions[0] 
    mask = cv2.resize(mask, (original_dimensions[0], original_dimensions[1]), interpolation=cv2.INTER_NEAREST)

    return mask

def parse_image_tf(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    return image

def generate_random_crop_boundary(image_shape, crop_size):
    max_y, max_x = image_shape[:2] - crop_size
    start_x = random.randint(0, max_x)
    start_y = random.randint(0, max_y)
    return start_x, start_y

def check_tissue_content(mask, crop_start_x, crop_start_y, crop_dim = 224, threshold =0.5):
    crop_mask = mask[crop_start_x:crop_start_x+crop_dim, crop_start_y:crop_start_y+crop_dim]
    
    tissue_pixels = np.sum(crop_mask > 0)
    total_pixels = crop_mask.size

    if (tissue_pixels / total_pixels) >= threshold:
        test_result = True
    else:
        test_result = False
    return test_result



def get_crops_bounds(mask, crop_dim = 224, max_attemopts = 10):

    crop_start_x, crop_start_y = generate_random_crop_boundary(mask.shape, crop_dim)
    
    crop_test_pass = check_tissue_content(mask, crop_start_x, crop_start_y)
    
    crop_attempt = 0
    while crop_attempt < max_attemopts:
        if crop_test_pass:
            crop_mask = mask[crop_start_x:crop_start_x+crop_dim, crop_start_y:crop_start_y+crop_dim]
            return crop_mask
        else:
            crop_attempt +1

    return None



def get_crops(filename, region_dim = 1024, crop_dim = 224):
    wsi_mask = parse_image4mask(filename)
    wsi_img = parse_image_tf(filename)

    valid_patch = False
    crops = []
    while not valid_patch:
        region_start_x, region_start_y = generate_random_crop_boundary(wsi_img.shape, region_dim)
        region_mask = wsi_mask[region_start_x:region_start_x+region_dim, region_start_y:region_start_y+region_dim]
        region_img = tf.image.crop_to_bounding_box(
                wsi_img,
                offset_height=region_start_y,
                offset_width=region_start_x,
                target_height=region_dim,
                target_width=region_dim
            )
        
        valid_crop_count = 0
        crop_bounds = get_crops_bounds(region_mask, crop_dim) # generates random boundery and tests tissue content. If no tissue content, returns None
        if crop_bounds is not None:
            img_crop = region_img[crop_bounds[0]:crop_bounds[1], crop_bounds[2]:crop_bounds[3]]
            valid_crop_count += 1
            crops.append(img_crop)
            if valid_crop_count == 2:
                valid_patch = True

    return crops

def simsiam_augmenter(img):

    # The following transforms expect the data to be [0, 1]
    img /= 255.0

    # random color jitter
    def _jitter_transform(x):
        return tfsim.augmenters.augmentation_utils.color_jitter.color_jitter_rand(
            x,
            np.random.uniform(0.0, 0.4), #brightness
            np.random.uniform(0.0, 0.4), #contrast
            np.random.uniform(0.0, 0.4), #saturation
            np.random.uniform(0.0, 0.1), #hue
            "multiplicative",
        )

    img = tfsim.augmenters.augmentation_utils.random_apply.random_apply(_jitter_transform, p=0.8, x=img)

    # # random grayscale
    def _grascayle_transform(x):
        return tfsim.augmenters.augmentation_utils.color_jitter.to_grayscale(x)

    img = tfsim.augmenters.augmentation_utils.random_apply.random_apply(_grascayle_transform, p=0.2, x=img)


    img = tfsim.augmenters.augmentation_utils.blur.random_blur(img, height= img.shape[0], width =img.shape[1], min_sigma = 0.2, max_sigma= 1.2, p=0.5)

    # random horizontal flip
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    # scale the data back to [0, 255]
    img = img * 255.0
    img = tf.clip_by_value(img, 0.0, 255.0)

    return img


@tf.function
def process(filename):
    # wsi to crops
    
    crops = get_crops(filename)
    #reduce augmenter to only transformations
    view1 = simsiam_augmenter(crops[0])
    # view1 = img_scaling(view1)
    view2 = simsiam_augmenter(crops[1])
    # view2 = img_scaling(view2)
    return view1, view2