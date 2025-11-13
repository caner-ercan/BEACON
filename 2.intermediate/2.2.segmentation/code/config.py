import os
import argparse

BE_MASTER = os.environ.get("BE_MASTER")
if not BE_MASTER:
    raise EnvironmentError("BE_MASTER environment variable must be set to the data root")

# Configuration settings
img_size = 512
batch_size = 8
num_epochs = 200
learning_rate = 1e-4
model_checkpoint = "nvidia/mit-b0"
output_folder = os.path.join(BE_MASTER, "2.intermediate/2.3.segmentation/training/finetuning/from_tcga_ds1")
split_csv = os.path.join(BE_MASTER, "2.intermediate/2.0.input/subset_list.csv")
tile_export_folder = os.path.join(BE_MASTER, "2.intermediate/2.3.segmentation/qp_proj_annotation/1024px_05mpp/segmentation_tiles")



# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--lr", type=float, default="1e-4")
parser.add_argument("--weight_path", type=str, default=None)
args = parser.parse_args()

exp_folder = os.path.join(output_folder, args.model_name.split("/")[-1], str(args.lr))
img_save_path = os.path.join(exp_folder, "plots")
os.makedirs(exp_folder, exist_ok=True)
os.makedirs(img_save_path, exist_ok=True)


#print configs
print("Model checkpoint: ", model_checkpoint)
print("Output folder: ", output_folder)
print("Split csv: ", split_csv)
print("Tile export folder: ", tile_export_folder)
print("Experiment folder: ", exp_folder)
print("Image save path: ", img_save_path)

print("Learning rate: ", args.lr)
print("Weight path: ", args.weight_path)
