import tensorflow as tf
import numpy as np
import tensorflow_similarity as tfsim


import albumentations as A

def simsiam_augmenter(img):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [224, 224])
    img = img.numpy()

    transform = A.Compose([
        # A.RandomResizedCrop(224, 224, scale=(0.2, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.ToGray(p=0.05),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    ])

    augmented = transform(image=img)['image']
    augmented = tf.convert_to_tensor(augmented, dtype=tf.float32)
    augmented = tf.clip_by_value(augmented, 0.0, 255.0)

    return augmented


# def simsiam_augmenter(img):
#     img = tf.cast(img, tf.float32)
#     img = tf.image.resize(img, [224, 224])
#     img /= 255.0

#     def _jitter_transform(x):
#         return tfsim.augmenters.augmentation_utils.color_jitter.color_jitter_rand(
#             x,
#             np.random.uniform(0.0, 0.4),
#             np.random.uniform(0.0, 0.4),
#             np.random.uniform(0.0, 0.4),
#             np.random.uniform(0.0, 0.1),
#             "multiplicative",
#         )

#     try:
#         img = tfsim.augmenters.augmentation_utils.random_apply.random_apply(_jitter_transform, p=0.8, x=img)
#     except Exception as e:
#         pass

#     def _grascayle_transform(x):
#         return tfsim.augmenters.augmentation_utils.color_jitter.to_grayscale(x)

#     try:
#         img = tfsim.augmenters.augmentation_utils.random_apply.random_apply(_grascayle_transform, p=0.05, x=img)
#     except Exception as e:
#         pass

#     try:
#         img = tfsim.augmenters.augmentation_utils.blur.random_blur(img, height=7, width=7, min_sigma=0.2, max_sigma=1.2, p=0.5)
#     except Exception as e:
#         pass

#     img = tf.image.random_flip_left_right(img)
#     img = tf.image.random_flip_up_down(img)
#     img = img * 255.0
#     img = tf.clip_by_value(img, 0.0, 255.0)

#     return img