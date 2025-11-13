import numpy as np
import tensorflow as tf
import random as rn
import os
from config import *
from data_loader import load_data, CustomImageDataGenerator, RandomAugmentation
from model import *
from utils import display, class_specific_iou, plot_images
import albumentations as A
from keras.callbacks import EarlyStopping


# Set seeds for reproducibility
np.random.seed(2023)
tf.random.set_seed(2023)
rn.seed(2023)

# Load data
x_train_dir, y_train_dir, x_val_dir, y_val_dir, x_test_dir, y_test_dir = load_data(tile_export_folder, split_csv)

# Data generators

random_augmentation = RandomAugmentation(p=0.5)

train_data_generator = CustomImageDataGenerator(x_df=x_train_dir, y_df=y_train_dir, batch_size=batch_size, img_size=img_size, augment=True, shuffle=True, transforms=random_augmentation)
valid_data_generator = CustomImageDataGenerator(x_df=x_val_dir, y_df=y_val_dir, batch_size=batch_size, img_size=img_size, augment=False, shuffle=False)
test_data_generator = CustomImageDataGenerator(x_df=x_test_dir, y_df=y_test_dir, batch_size=batch_size, img_size=img_size, augment=False, shuffle=False)

# Create and train model
model = create_model(args.model_name, args.lr, args.weight_path)
callbacks = [EarlyStopping(monitor='val_loss', patience=40, verbose=1, restore_best_weights=True)]
history = train_model(model, train_data_generator, valid_data_generator, int(np.ceil(len(x_train_dir) / batch_size)), num_epochs, callbacks)

# Save model
model.save_weights(os.path.join(exp_folder, "finetuned_model.h5"))

# Evaluate model
train_data_generator.augment = False
train_data_generator.shuffle = False
train_pred = class_specific_iou(train_data_generator, y_train_dir, model, exp_folder, img_save_path, "train", id2label)
val_pred = class_specific_iou(valid_data_generator, y_val_dir, model, exp_folder, img_save_path, "val", id2label)
test_pred = class_specific_iou(test_data_generator, y_test_dir, model, exp_folder, img_save_path, "test", id2label)

# Plot results
plot_images(x_train_dir, y_train_dir, train_pred, img_save_path, "train")
plot_images(x_val_dir, y_val_dir, val_pred, img_save_path, "val")
plot_images(x_test_dir, y_test_dir, test_pred, img_save_path, "test")
