import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

def mask_to_rgb(mask, class_color_match):
    mask = np.asarray(mask).astype(np.uint8)
    image_size = mask.shape[1]
    rgb_mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    for h in range(image_size):
        for w in range(image_size):
            rgb_mask[h, w] = class_color_match.get(int(mask[h, w]))

    return rgb_mask

def display(display_list, img_save_path):
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "True Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        if i == 1 or i == 2:
            rgb_array = mask_to_rgb(display_list[i], class_color_match)
            plt.imshow(tf.keras.utils.array_to_img(rgb_array))
        else:
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")

    plt.savefig(os.path.join(img_save_path, "plot.png"))

def calculate_iou(y_true, y_pred, num_classes=4):
    iou_scores = []
    for class_id in range(num_classes):
        intersection = np.sum((y_pred == class_id) & (y_true == class_id))
        union = np.sum((y_pred == class_id) | (y_true == class_id))
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)
    return iou_scores

def class_specific_iou(data_generator, y_dir_array, model, exp_folder, img_save_path, ds, id2label):
    output = model.predict(data_generator)
    predicted_labels = np.argmax(output.logits, axis=1)
    predicted_labels_resized = np.zeros((predicted_labels.shape[0], 512, 512), dtype=np.uint8)

    for i in range(predicted_labels.shape[0]):
        img = Image.fromarray(predicted_labels[i].astype(np.uint8))
        img = img.resize((512, 512), Image.NEAREST)
        predicted_labels_resized[i] = np.array(img)

    y = []
    for f in y_dir_array:
        img = Image.open(f).resize((512, 512), Image.NEAREST)
        y.append(np.array(img))
    y = np.array(y)

    iou_scores = calculate_iou(y, predicted_labels_resized)


    cm = confusion_matrix(y.flatten(), predicted_labels_resized.flatten(), normalize='true')
    acc_class =[]
    for i in range(len(cm)):
        acc_class.append(cm[i][i])


    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.1%', cmap='Blues', xticklabels=id2label.values(), yticklabels=id2label.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(img_save_path, f"{ds}_cm.png"))


    columns = ['path', 'dataset']
    for class_id in id2label.keys():
        class_name = id2label.get(class_id, f"class_{class_id}")
        columns.append(f'{class_name}_iou')
        columns.append(f'{class_name}_acc')

    data = pd.DataFrame(columns=columns)

    # Append the initial values to the DataFrame
    data.loc[0] = [exp_folder, ds] + [None] * (len(columns) - 2)
    
    for class_id, iou in enumerate(iou_scores):
        class_name = id2label.get(class_id, f"class_{class_id}")
        data.at[0, f'{class_name}_iou'] = iou
        data.at[0, f'{class_name}_acc'] = acc_class[class_id]

    csv_path = os.path.join(exp_folder, f"results.csv")
    if os.path.exists(csv_path):
        data.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        data.to_csv(csv_path, mode='w', header=True, index=False)

    return predicted_labels_resized

def plot_images(x_val_dir, y_val_dir, predicted_labels_resized, img_save_path, ds, num_images=10):
    plt.figure(figsize=(15, 5 * num_images))
    x_val = [np.array(Image.open(f)) for f in x_val_dir]
    y_val = [np.array(Image.open(f)) for f in y_val_dir]

    for i in range(num_images):
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(x_val[i])
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(y_val[i], cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')

        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(predicted_labels_resized[i], cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(img_save_path, f"{ds}_pred.png"))
