import os
import pandas as pd
from PIL import Image
import numpy as np
import albumentations as A
from tensorflow.keras.utils import Sequence
import random



class RandomAugmentation:
    def __init__(self, p=0.5):
        self.aug = A.Compose([
            A.RandomBrightnessContrast(p=0.45),
            A.HueSaturationValue(p=0.45),
            A.GaussNoise(p=0.2),
            A.Blur(p=0.2),
        ])
        self.p = p

    def __call__(self, *, image):
        # always return a dict with 'image' key
        if random.random() < self.p:
            return self.aug(image=image)       # this is already a dict
        else:
            return {"image": image}

class CustomImageDataGenerator(Sequence):
    def __init__(self, x_df, y_df, batch_size=8, img_size=None, augment=False, shuffle=True, transforms=None, random_state=42):
        self.x_df = x_df.copy()
        self.y_df = y_df.copy()
        self.augment = augment
        self.img_size = img_size
        self.batch_size = batch_size
        self.n = len(self.x_df)
        self.indexes = np.arange(self.n)
        self.transform = transforms
        self.shuffle = shuffle
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_images = []
        batch_masks = []

        for i in batch_indexes:
            image_path = self.x_df[i]
            image = Image.open(image_path).resize((self.img_size, self.img_size))

            if self.augment:
                image = self.transform(image=np.array(image))['image']

            image_array = np.array(image) / 255.0
            image_array = np.transpose(image_array, (2, 0, 1))
            batch_images.append(image_array)

            mask_path = self.y_df[i]
            mask = Image.open(mask_path).resize((self.img_size, self.img_size))
            batch_masks.append(np.array(mask))

        return np.array(batch_images), np.array(batch_masks)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

def load_data(tile_export_folder, split_csv):
    split_df = pd.read_csv(split_csv)
    x_train_dir, y_train_dir, x_val_dir, y_val_dir, x_test_dir, y_test_dir = [], [], [], [], [], []

    for roi in [f for f in os.listdir(tile_export_folder) if f.startswith('D')]:
        mask_files = [f for f in os.listdir(os.path.join(tile_export_folder, roi)) if f.endswith(']-labelled.png')]
        tile_files = [f for f in os.listdir(os.path.join(tile_export_folder, roi)) if f.endswith('].png')]

        split = split_df.loc[split_df['slide_id'].str.contains(roi[:5])]["subset"].values[0]

        if split == 'train':
            x_train_dir.extend([os.path.join(tile_export_folder, roi, f) for f in tile_files])
            y_train_dir.extend([os.path.join(tile_export_folder, roi, f) for f in mask_files])
        elif split == 'val':
            x_val_dir.extend([os.path.join(tile_export_folder, roi, f) for f in tile_files])
            y_val_dir.extend([os.path.join(tile_export_folder, roi, f) for f in mask_files])
        elif split == 'test':
            x_test_dir.extend([os.path.join(tile_export_folder, roi, f) for f in tile_files])
            y_test_dir.extend([os.path.join(tile_export_folder, roi, f) for f in mask_files])

    return sorted(x_train_dir), sorted(y_train_dir), sorted(x_val_dir), sorted(y_val_dir), sorted(x_test_dir), sorted(y_test_dir)
