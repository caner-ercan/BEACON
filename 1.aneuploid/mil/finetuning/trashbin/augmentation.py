import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import AveragePooling2D, Dense, Flatten
import time
from tqdm import tqdm
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

DEVICE = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'

def get_complete_transform(output_shape, kernel_size, s=1.0):
    """
    Color distortion transform

    Args:
        s: Strength parameter

    Returns:
        A color distortion transform
    """
    rnd_crop = A.RandomResizedCrop(output_shape[0], output_shape[1], scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
    rnd_flip = A.HorizontalFlip(p=0.5)

    color_jitter = A.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s, p=0.8)
    rnd_color_jitter = A.Compose([color_jitter], p=0.8)

    rnd_gray = A.ToGray(p=0.2)
    gaussian_blur = A.GaussianBlur(blur_limit=(3, 7), p=0.5)
    rnd_gaussian_blur = A.Compose([gaussian_blur], p=0.5)
    to_tensor = ToTensorV2()
    image_transform = A.Compose([
        to_tensor,
        rnd_crop,
        rnd_flip,
        rnd_color_jitter,
        rnd_gray,
        rnd_gaussian_blur,
    ])
    return image_transform


class ContrastiveLearningViewGenerator(object):
    """
    Take 2 random crops of 1 image as the query and key.
    """
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views
        
    def __call__(self, x):
        views = [self.base_transform(image=x)['image'] for i in range(self.n_views)]
        return views

class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, list_images, transform=None):
        """
        Args:
            list_images (list): List of all the images
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.list_images = list_images
        self.transform = transform
        
    def __len__(self):
        return len(self.list_images)
    
    def __getitem__(self, idx):
        if tf.is_tensor(idx):
            idx = idx.numpy()

        img_name = self.list_images[idx]
        image = cv2.imread(img_name) # this will be replaced with following:
        # h5file is read for image (the patching is done with 1024 x 1024 patches)
        # patch is selected from h5file coords list
        # openslide reads the region
        # or
        # the patches will be exracted to a folder

        if self.transform:
            image = self.transform(image=image)

        return image

# Dataset
out_shape = [224, 224]
kernel_size = [21, 21] # 10% of out_shape

# Custom transform
base_transforms = get_complete_transform(output_shape=out_shape, kernel_size=kernel_size, s=1.0)
custom_transform = ContrastiveLearningViewGenerator(base_transform=base_transforms)

# Define the path to the directory containing the .tif files
base_path1 = "/Volumes/rsrch5/home/trans_mol_path/cercan/data/segmentation/wholeDataset_inference/241119/MDA1_dataset_rois/tiles"
base_path2 = "/Volumes/rsrch5/home/trans_mol_path/cercan/data/segmentation/wholeDataset_inference/241119/MDA2_dataset_rois/tiles"

# Use glob to find all .tif files in all subfolders
tif_files1 = glob.glob(f"{base_path1}/**/*.png", recursive=True)
tif_files2 = glob.glob(f"{base_path2}/**/*.png", recursive=True)
tif_files = tif_files1 + tif_files2

# Create the CustomDataset with the list of .tif files and the custom transform
deneme_ds = CustomDataset(
    list_images=tif_files,
    transform=custom_transform
)

BATCH_SZ = 64

foundation_model_base = '/rsrch5/home/trans_mol_path/cercan/foundationModels/physionet.org/files/medical-ai-research-foundation/1.0.0/'
def foundationModel(arch = 'path-50x1-remedis-s',no_linear=False, noAverage = False):
    device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
    with tf.device(device):
        hub_path = os.path.join(foundation_model_base,arch)

        module_keras=hub.KerasLayer(hub_path, trainable=True)
        model = tf.keras.Sequential(module_keras)
        model.build(input_shape=(None, 224,224,3))
        #model.add(Flatten()) # avrpool
        if not noAverage:
            model.add(AveragePooling2D(pool_size=(7, 7))) #TODO change to trainable conv layer
            model.add(Flatten())
        if not no_linear:
            model.add(Dense(1024, activation= None,use_bias=False)) # input_shape=(131072,)

    return model

class Identity(tf.keras.layers.Layer):
    def call(self, x):
        return x

class SimCLR(tf.keras.Model):
    def __init__(self, linear_eval=False, arch='path-50x1-remedis-s'):
        super().__init__()
        self.linear_eval = linear_eval
        self.arch = arch
        # resnet18 = models.resnet18(pretrained=False)
        remedis_model = foundationModel(arch = self.arch)
        remedis_model.fc = Identity()
        self.encoder = remedis_model
        self.projection = tf.keras.Sequential([
            tf.keras.layers.Dense(512),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(256)
        ])

    def call(self, x):
        if not self.linear_eval:
            x = tf.concat(x, axis=0)
        encoding = self.encoder(x)
        projection = self.projection(encoding)
        return projection

# Contrastive Loss

LABELS = tf.concat([tf.range(BATCH_SZ) for i in range(2)], axis=0)
LABELS = tf.cast(tf.equal(tf.expand_dims(LABELS, axis=0), tf.expand_dims(LABELS, axis=1)), tf.float32) #one-hot representations
LABELS = tf.cast(LABELS, tf.float32)

def ntxent_loss(features, temp):
    """
    NT-Xent Loss.

    Args:
        z1: The learned representations from first branch of projection head
        z2: The learned representations from second branch of projection head
    Returns:
        Loss
    """
    similarity_matrix = tf.matmul(features, features, transpose_b=True)
    mask = tf.eye(tf.shape(LABELS)[0], dtype=tf.bool)
    labels = tf.reshape(LABELS[~mask], (tf.shape(LABELS)[0], -1))
    similarity_matrix = tf.reshape(similarity_matrix[~mask], (tf.shape(similarity_matrix)[0], -1))

    positives = tf.reshape(similarity_matrix[tf.cast(labels, tf.bool)], (tf.shape(labels)[0], -1))

    negatives = tf.reshape(similarity_matrix[~tf.cast(labels, tf.bool)], (tf.shape(similarity_matrix)[0], -1))

    logits = tf.concat([positives, negatives], axis=1)
    labels = tf.zeros(tf.shape(logits)[0], dtype=tf.int64)

    logits = logits / temp
    return logits, labels

import matplotlib.pyplot as plt

# ... (rest of your code remains the same)

# Train
simclr_model = SimCLR()
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

epochs = 1000
loss_values = []  # List to store loss values

# with tqdm(total=epochs) as pbar:
for epoch in range(epochs):
    # t0 = time.time()
    running_loss = 0.0
    for i, views in enumerate(deneme_ds):
        with tf.GradientTape() as tape:
            with tf.device(DEVICE):
                projections = simclr_model([view for view in views])
                logits, labels = ntxent_loss(projections, temp=2)
                loss = criterion(logits, labels)
        gradients = tape.gradient(loss, simclr_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, simclr_model.trainable_variables))

        # print stats
        running_loss += loss.numpy()
        if i % 10 == 9: # print every 10 mini-batches
            print(f"Epoch: {epoch+1} Batch: {i+1} Loss: {(running_loss/100):.4f}")
            running_loss = 0.0
        # pbar.update(1)
        # print(f"Time taken: {((time.time()-t0)/60):.3f} mins")

    # Collect the average loss for the epoch
    epoch_loss = running_loss / len(deneme_ds)
    loss_values.append(epoch_loss)

# Plot the loss values
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
