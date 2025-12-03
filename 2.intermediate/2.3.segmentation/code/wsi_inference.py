import numpy as np
import tensorflow as tf
import os
import random as rn
from pandas import DataFrame
from sklearn.metrics import multilabel_confusion_matrix
from PIL import Image
from transformers import TFAutoModelForSemanticSegmentation
from tqdm import tqdm
from tensorflow.keras.utils import Sequence


id2label = {
    0: 'background',
    1: 'columnar',
    2: 'stroma',
    3: 'squamous',
}
label2id = {label: id for id, label in id2label.items()}
nClasses = len(id2label)


cache_dir = "/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/finetuner/model_cache"

id2label_tcga = {
    0: 'background',
    1: 'tumor',
    2: 'stroma',
    3: 'inflam',
    4: 'necrosis',
    5: 'fat',
    6: 'parenchyma',
    7: 'blood_vessel',
    8: 'blood'
}
label2id_tcga = {label: id for id, label in id2label_tcga.items()}
num_labels_tcga = len(id2label_tcga)

def create_model(model_checkpoint, weight_path=None ):
    if not weight_path:
        model = TFAutoModelForSemanticSegmentation.from_pretrained(
            model_checkpoint,
            num_labels=nClasses,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            cache_dir=cache_dir
        )
        model.compile(optimizer=Adam(learning_rate=learning_rate))
    else:
        model = TFAutoModelForSemanticSegmentation.from_pretrained(
            model_checkpoint,
            num_labels=num_labels_tcga,
            id2label=id2label_tcga,
            label2id=label2id_tcga,
            ignore_mismatched_sizes=True,
            cache_dir=cache_dir
        )
        # model.load_weights(os.path.join(weight_path, "tf_model.h5"))
        model.load_weights(model_checkpoint)

        # base_model_4c = TFAutoModelForSemanticSegmentation.from_pretrained(
        #     model_checkpoint,
        #     num_labels=nClasses,
        #     id2label=id2label,
        #     label2id=label2id,
        #     ignore_mismatched_sizes=True,
        #     cache_dir=cache_dir
        # )
        # new_decode = base_model_4c.decode_head
        # model.decode_head = new_decode

        # model.config.id2label = id2label
        # model.config.label2id = label2id
        # model.config.num_labels = nClasses

        print("Weights loaded from:", weight_path)

    # model.compile(optimizer=Adam(learning_rate=learning_rate))
    return model

model_path='/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/training/finetuning/from_tcga_ds1/mit-b0/1e-05/finetuned_model.h5'

model = TFAutoModelForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=nClasses,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            cache_dir=cache_dir
        )
model.load_weights(model_path)
model.summary()

# input_dir = '/rsrch5/home/trans_mol_path/xpan7/tmeseg/patch768_tcga20All/image'
# target_dir = '/rsrch5/home/trans_mol_path/xpan7/tmeseg/patch768_tcga20All/maskPng'
img_size = (1024, 1024)
image_size = 1024
nClasses = 4
batch_size = 64
auto = tf.data.AUTOTUNE


# tiles_folder="/rsrch5/home/trans_mol_path/cercan/data/clam/segmentation/wsi_tiles"
tiles_folder="/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.4.inference/tiles_segmentation/"
wsi_samples = [filename for filename in os.listdir(tiles_folder) if os.path.isdir(os.path.join(tiles_folder,filename))]
output_folder =  "/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.4.inference/wsi_segment_masks/"

# single_sample = [wsi_samples[0]]


def get_paths(tiles_folder,sample_list,file_type):
    all_tiles_paths = []
    for sample in sample_list:
        folder=os.path.join(tiles_folder,sample)
        single_WSI_files = [file for file in os.listdir(folder) if file.endswith(file_type)]
        single_WSI_files = [item for item in single_WSI_files if not item.startswith('.')]
        single_WSI_files = [os.path.join(folder, file) for file in single_WSI_files]
        all_tiles_paths.extend(single_WSI_files)
    return all_tiles_paths

# def get_all_paths(tiles_folder, list_NU, list_MDA):
#     x_dir_NU = get_paths(tiles_folder, "NU",list_NU,"].png")
#     x_dir_MDA = get_paths(tiles_folder, "MDA",list_MDA,"].png")
#     x_dir = x_dir_NU + x_dir_MDA
    
#     # y_dir_NU = get_paths(tiles_folder, "NU",list_NU,".png")
#     # y_dir_MDA = get_paths(tiles_folder, "MDA",list_MDA,".png")
#     # y_dir = y_dir_NU + y_dir_MDA
#     y_dir=[]
#     for x_img in x_dir:
#         y_img = x_img.replace("].png", "]-labelled.png")
#         y_dir.append(str(y_img))
    
#     for i in range(len(x_dir)):
#         if x_dir[i][:-4] != y_dir[i][:-13]:
#             print("Mismatch:")
#             print(x_dir[i])
#             print(y_dir[i])
    
#     # x_dir = x_dir.sort
#     # y_dir = y_dir.sort
#     return x_dir, y_dir

all_tiles_paths = get_paths(tiles_folder, wsi_samples,".png")


test_samples = len(all_tiles_paths)
print("Test sample patches: {}".format(test_samples))


# df_test = DataFrame(all_tiles_paths,columns=['filename'])

def read_image(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def load_img_val(img):
    image = read_image(img)
    image /= 255.0
    return image
def categorize_images(image_paths, target_size=(1024, 1024)):
    typical_images = []
    atypical_images = []

    for img_path in image_paths:
        img = load_img_val(img_path)
        if img.shape[:2] == list(target_size):
            typical_images.append(img_path)
        else:
            atypical_images.append(img_path)

    return typical_images, atypical_images

# # Example usage:
 # Your list of image paths
typical_images, atypical_images = categorize_images(all_tiles_paths)

class InferenceImageDataGenerator(Sequence):
    def __init__(self, x_df, batch_size=8, img_size=512, target_path=None, pad_to_size=False):
        self.x_df = x_df.copy()
        self.batch_size = batch_size
        self.img_size = img_size
        self.n = len(self.x_df)
        self.indexes = np.arange(self.n)
        self.target_path = target_path
        self.pad_to_size = pad_to_size

        # Ensure the target directory exists
        if self.target_path and not os.path.exists(self.target_path):
            os.makedirs(self.target_path)

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_images = []
        batch_filenames = []
        original_sizes = []

        for i in batch_indexes:
            image_path = self.x_df[i]
            image = Image.open(image_path)
            if self.pad_to_size:
                original_size = image.size  # Store original size for later cropping
                original_sizes.append(original_size)
                # Pad image to (img_size, img_size) if it's smaller
                image = self.pad_image(image)
            
            image = Image.open(image_path)

            
            image = image.resize((self.img_size, self.img_size))
            image_array = np.array(image) / 255.0
            image_array = np.transpose(image_array, (2, 0, 1))
            batch_images.append(image_array)
            batch_filenames.append(os.path.basename(image_path))

        return np.array(batch_images), batch_filenames, original_sizes

    def pad_image(self, image):
        # Create a new white image of size (img_size, img_size)
        padded_image = Image.new("RGB", (self.img_size, self.img_size), (255, 255, 255))
        # Paste the original image in the center
        offset = ((self.img_size - image.width) // 2, (self.img_size - image.height) // 2)
        padded_image.paste(image, offset)
        return padded_image

    def crop_to_original_size(self, prediction, original_size):
        # Crop the prediction back to the original size
        pred_image = Image.fromarray((prediction * 255).astype(np.uint8))
        offset = ((self.img_size - original_size[0]) // 2, (self.img_size - original_size[1]) // 2)
        cropped_image = pred_image.crop((offset[0], offset[1], offset[0] + original_size[0], offset[1] + original_size[1]))
        return cropped_image

    def save_predictions(self, predictions, filenames, original_sizes=None):
        if not self.target_path:
            raise ValueError("Target path is not set.")

        for i, (pred, filename) in enumerate(zip(predictions, filenames)):
            pred = Image.fromarray(pred.astype(np.uint8))
            if self.pad_to_size:
                pred = self.crop_to_original_size(pred, original_sizes[i])
            
            pred = pred.resize((self.img_size,self.img_size), Image.NEAREST)
            
            pred.save(os.path.join(self.target_path, filename))

    def on_epoch_end(self):
        # No shuffling needed for inference
        pass


typical_img_gen = InferenceImageDataGenerator(typical_images, batch_size=128, img_size=512, target_path=output_folder)
for images, filenames, _ in typical_img_gen:
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions.logits, axis=1)
    typical_img_gen.save_predictions(predicted_labels, filenames)

#the partial tiles
atypical_img_gen = InferenceImageDataGenerator(atypical_images, batch_size=128, img_size=512, target_path=output_folder,pad_to_size=True)
for images, filenames, original_sizes in atypical_img_gen:
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions.logits, axis=1)
    typical_img_gen.save_predictions(predicted_labels, filenames, original_sizes)

    