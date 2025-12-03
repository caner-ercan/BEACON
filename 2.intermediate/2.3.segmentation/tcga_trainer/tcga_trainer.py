import numpy as np
import tensorflow as tf
import random as rn
import os
from glob import glob
from tensorflow.keras import backend
from transformers import create_optimizer
from transformers import TFAutoModelForSemanticSegmentation
from transformers import DefaultDataCollator
from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback
from transformers import AutoImageProcessor
from tensorflow.keras.optimizers.legacy import Adam
from keras.callbacks import EarlyStopping
import argparse

np.random.seed(2023)
tf.random.set_seed(2023)
rn.seed(2023)
tf.keras.utils.set_random_seed(2023)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="nvidia/mit-b3")
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()


# source_path = "/rsrch6/home/trans_mol_path/yuan_lab/TIER2/barrett/tme/4Caner_patch896_BCSS/"
source_path = "/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/finetuner/tcga_trainer/4Caner_patch896_BCSS/"
img = glob(os.path.join(source_path, 'image/*.png')) # tf.io.glob.glob

img_filenames = [path.split('/image/')[1][:-4] for path in img]

slide_names = [name.split('_')[0] for name in img_filenames]
unique_slide_names = list(set(slide_names))
train_slide_names = unique_slide_names[:int(0.8*len(unique_slide_names))]
val_slide_names = unique_slide_names[int(0.8*len(unique_slide_names)):]

train_img_filenames = []
for train_slide_name in train_slide_names:
    train_img_names_slide = [name for name in img_filenames if name.startswith(train_slide_name)]
    train_img_filenames.extend(train_img_names_slide)
train_img_filenames
train_img_paths = sorted([os.path.join(source_path, 'image', name + '.png') for name in train_img_filenames])
train_label_paths = sorted([os.path.join(source_path, 'maskPng', 'mask_' + name + '.png') for name in train_img_filenames])

val_img_filenames = []
for val_slide_name in val_slide_names:
    val_img_names_slide = [name for name in img_filenames if name.startswith(val_slide_name)]
    val_img_filenames.extend(val_img_names_slide)

val_img_paths = sorted([os.path.join(source_path, 'image', name + '.png') for name in val_img_filenames])
val_label_paths = sorted([os.path.join(source_path, 'maskPng', 'mask_' + name + '.png') for name in val_img_filenames])

print("Number of train slides: {}, tiles: {}".format(len(train_slide_names),len(train_img_filenames)))
print("Number of valid slides: {}, tiles: {}".format(len(val_slide_names),len(val_img_filenames)))

image_size = 512
learning_rate = args.lr #0.0001 #TO-Do arg for lr
auto = tf.data.AUTOTUNE
batch_size = 8
# nClasses = 8
num_epochs = 80
num_img = len(train_img_filenames)


def read_image(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [image_size, image_size])
    return img

def read_mask(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [image_size, image_size])
    return img

#######################
def rand_crop(img, mask):
    concat_img = tf.concat([img, mask], axis=-1)
    concat_img = tf.image.resize(concat_img, [280, 560], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    crop_img = tf.image.random_crop(concat_img, [256, 256, 4])
    return crop_img[:, :, :3], crop_img[:, :, 3:]

def norm(image, mask):
    image = tf.cast(image, tf.float32) / 0.5 - 1
    #mask = tf.cast(mask, tf.int32)
    return image, mask

def normalize(input_image, input_mask):
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    #input_mask /= 255. mutli class no need this step
    input_mask = tf.image.convert_image_dtype(input_mask, tf.int32)
    return input_image, input_mask
#######################



def aug_transforms(image):
    image /= 255.0
    image = tf.image.random_brightness(image, 0.25)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.8, 2.0)
    image = tf.image.random_hue(image, 0.1)
    return image

def load_img_train(img, mask):
    image = read_image(img)
    mask = read_mask(mask)
    image = aug_transforms(image)
    #image, mask = norm(image, mask)
    image = tf.transpose(image, (2, 0, 1)) #for segformer
    mask = tf.squeeze(mask, axis=-1)
    # print(image.shape)
    # print(mask.shape)
    return image, mask

def load_img_val(img, mask):
    image = read_image(img)
    mask = read_mask(mask)
    image = tf.transpose(image, (2, 0, 1))
    mask = tf.squeeze(mask, axis=-1)
    # print(image.shape)
    # print(mask.shape)
    return normalize(image, mask)


train_ds = tf.data.Dataset.from_tensor_slices((train_img_paths, train_label_paths))
val_ds = tf.data.Dataset.from_tensor_slices((val_img_paths, val_label_paths))

train_ds = train_ds.map(load_img_train, num_parallel_calls=auto)
train_ds = train_ds.repeat()
train_ds = train_ds.shuffle(20).batch(batch_size)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(load_img_val, num_parallel_calls=auto)
# val_ds = val_ds.repeat()
val_ds = val_ds.shuffle(20).batch(batch_size)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

id2label = {
    0:  'background',
    1:  'tumor',
    2:  'stroma',
    3:  'inflam',
    4:  'necrosis',
    5:  'fat',
    6:  'parenchyma',
    7:  'blood_vessel',
    8: 'blood'
} #8:  'alveoli'
label2id = {label: id for id, label in id2label.items()}
num_labels = len(id2label)


# model_checkpoint = "nvidia/mit-b3"
#image_processor = AutoImageProcessor.from_pretrained(model_checkpoint), from scratc
optimizer = Adam(learning_rate=learning_rate)
model = TFAutoModelForSemanticSegmentation.from_pretrained(args.model_name, num_labels=num_labels,
                                                           id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True,
        cache_dir = "/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/finetuner/model_cache"
    )
model.compile(optimizer=optimizer)
callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)]


output_folder = "/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/training/finetuning/tcga_pretrain"

exp_folder = os.path.join(output_folder, args.model_name.split("/")[-1]+ "_tcga",str(learning_rate))


model.fit(train_ds, validation_data = val_ds,
          steps_per_epoch=int(np.ceil(len(train_img_filenames) / batch_size)),
          epochs=num_epochs, verbose=1)

model.save_pretrained(exp_folder)

