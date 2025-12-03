import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from glob import glob
from tensorflow.keras.optimizers.legacy import Adam
from transformers import TFAutoModelForSemanticSegmentation
import argparse
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
import argparse

np.random.seed(2023)
tf.random.set_seed(2023)


source_path = "/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/finetuner/tcga_trainer/4Caner_patch896_BCSS/"
img = glob(os.path.join(source_path, 'image/*.png'))

img_filenames = [path.split('/image/')[1][:-4] for path in img]

slide_names = [name.split('_')[0] for name in img_filenames]
unique_slide_names = list(set(slide_names))
train_slide_names = unique_slide_names[:int(0.8*len(unique_slide_names))]
val_slide_names = unique_slide_names[int(0.8*len(unique_slide_names)):]

train_img_filenames = []
for train_slide_name in train_slide_names:
    train_img_names_slide = [name for name in img_filenames if name.startswith(train_slide_name)]
    train_img_filenames.extend(train_img_names_slide)
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
# learning_rate = lr
auto = tf.data.AUTOTUNE
batch_size = 8
num_epochs = 80

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

def load_img_val(img, mask):
    image = read_image(img)
    mask = read_mask(mask)
    image = tf.transpose(image, (2, 0, 1))
    mask = tf.squeeze(mask, axis=-1)
    return normalize(image, mask)

def normalize(input_image, input_mask):
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_mask = tf.image.convert_image_dtype(input_mask, tf.int32)
    return input_image, input_mask



train_ds = tf.data.Dataset.from_tensor_slices((train_img_paths, train_label_paths))
train_ds = train_ds.map(load_img_val, num_parallel_calls=auto)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)


val_ds = tf.data.Dataset.from_tensor_slices((val_img_paths, val_label_paths))
val_ds = val_ds.map(load_img_val, num_parallel_calls=auto)
val_ds = val_ds.batch(batch_size)
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
}
label2id = {label: id for id, label in id2label.items()}
num_labels = len(id2label)



train_y = []
for f in train_label_paths:
    img = Image.open(f)
    img = img.resize((512, 512), Image.NEAREST)  # Use nearest-neighbor interpolation for masks
    train_y.append(np.array(img))
train_y = np.array(train_y)


val_y = []
for f in val_label_paths:
    img = Image.open(f)
    img = img.resize((512, 512), Image.NEAREST)  # Use nearest-neighbor interpolation for masks
    val_y.append(np.array(img))
val_y = np.array(val_y)




def calculate_iou(y_true, y_pred, num_classes=9):
    iou_scores = []
    for class_id in range(num_classes):
        intersection = np.sum((y_pred == class_id) & (y_true == class_id))
        union = np.sum((y_pred == class_id) | (y_true == class_id))
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)
    return iou_scores

def class_specific_iou(train_generator, val_generator, train_y, val_y, model, id2label, metric_output_path_prefix):
    datasets = {
        "val": (val_generator, val_y, "Validation"),
        "train": (train_generator, train_y, "Training")
    }

    
    for dataset_type, (data_generator, y, dataset_name) in datasets.items():
        output = model.predict(data_generator)
        predicted_labels = np.argmax(output.logits, axis=1)
        predicted_labels_resized = np.zeros((predicted_labels.shape[0], 512, 512), dtype=np.uint8)

        for i in range(predicted_labels.shape[0]):
            img = Image.fromarray(predicted_labels[i].astype(np.uint8))
            img = img.resize((512, 512), Image.NEAREST)  # Use nearest-neighbor interpolation for labels
            predicted_labels_resized[i] = np.array(img)

        # y = []
        # for f in y_dir_array:
        #     img = Image.open(f)
        #     img = img.resize((512, 512), Image.NEAREST)  # Use nearest-neighbor interpolation for masks
        #     y.append(np.array(img))
        # y = np.array(y)

        # iou_scores = calculate_iou(y, predicted_labels_resized)
        results = {}
        results["exp"] = os.path.split(metric_output_path_prefix)[1]
        for class_id in range(len(id2label)):
            # add class specific iou to results df as a new column
            results[f"{dataset_name}_{id2label[class_id]}_iou"] = calculate_iou(y, predicted_labels_resized, class_id)
            # class specific acc from confusion matrix
            # results[f"{dataset_name}_{id2label[class_id]}_acc"] = confusion_matrix(y.flatten(), predicted_labels_resized.flatten(), normalize='true')[class_id, class_id]




        # results[dataset_name] = {
        #     "iou_scores": iou_scores,
        #     "confusion_matrix": confusion_matrix(y.flatten(), predicted_labels_resized.flatten(), normalize='true')
        # }
        

        # Print IoU scores for each class
        # for class_id, iou in enumerate(iou_scores):
        #     print(f"{dataset_name} - Class {class_id}: IoU = {iou:.4f}")

    # Plot and save confusion matrices
    # for dataset_name, result in results.items():
        cm_plot_path = f'{metric_output_path_prefix}_{dataset_name}_confusion_matrix.png'
        print(cm_plot_path)
        cm = confusion_matrix(y.flatten(), predicted_labels_resized.flatten(), normalize='true')
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.1%', cmap='Blues', xticklabels=id2label.values(), yticklabels=id2label.values())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {dataset_name}')
        plt.savefig( cm_plot_path)
        plt.close()
    print(results)
    metrics_df = pd.DataFrame(results)
    metrics_csv_path = f'{metric_output_path_prefix}_metrics.csv'
    print(metrics_csv_path)
    # Export metrics to CSV
    # if file does not exist create, if exists add as a new line
    if not os.path.exists(metrics_csv_path):
        metrics_df.to_csv(metrics_csv_path)
    else:
        # Open csv and add a new line
        metrics_df.to_csv(metrics_csv_path)


# parser = argparse.ArgumentParser()
# parser.add_argument("--model_name", type=str, default="nvidia/mit-b3")
# parser.add_argument("--lr", type=float, default=1e-3)
# args = parser.parse_args()
model_parent_folder = "/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/training/finetuning/tcga_pretrain"

pretrained_model_names = [f for f in os.listdir(model_parent_folder) if f.startswith("mit")]

metric_output_folder = os.path.join(model_parent_folder, "metrics")
os.makedirs(metric_output_folder, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_name", type=str, default="nvidia/mit-b3")
parser.add_argument("--lr", type=str, default=1e-3)
args = parser.parse_args()


base_model_name = args.base_model_name
pretrained_model_name = base_model_name.split("/")[-1]+ "_tcga"
learning_rate = args.lr

# for pretrained_model_name in pretrained_model_names:
#     base_model_name = "nvidia/{}".format(pretrained_model_name.split('_')[0])
#     for learning_rate in os.listdir(os.path.join(model_parent_folder, pretrained_model_name)):
        

model = TFAutoModelForSemanticSegmentation.from_pretrained(base_model_name, num_labels=num_labels,
                                                    id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True,
cache_dir = "/rsrch5/home/trans_mol_path/cercan/BE_master/2.intermediate/2.3.segmentation/finetuner/model_cache"
)
model_path = os.path.join(model_parent_folder, pretrained_model_name, learning_rate, 'tf_model.h5')
model.compile(optimizer=Adam(learning_rate=float(learning_rate)))
model.load_weights(model_path)
metric_output_path_prefix = f'{metric_output_folder}/{pretrained_model_name}_{learning_rate}'
class_specific_iou(train_ds, val_ds, train_y, val_y, model, id2label, metric_output_path_prefix)

