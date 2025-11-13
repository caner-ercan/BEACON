import os
import h5py
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', dest='feature_folder', type=str, default=None)
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



# seed_folder_names = os.listdir(args.feature_folder)
# for seed_folder_name in seed_folder_names:
#     seed_folder = os.path.join(args.feature_folder,seed_folder_name)
seed_folder = args.feature_folder
#folder_list =["h5", "pt"]
#for subfolder in folder_list:

data_folder = os.path.join(seed_folder,"h5_files")
output_folder = os.path.join(seed_folder,"merged","h5_files")
print("writing features to", output_folder)
output_folder_pt = os.path.join(seed_folder,"merged","pt_files")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder_pt, exist_ok=True)
filenames = [filename for filename in os.listdir(data_folder) if filename.endswith("ROI0.h5")]

core_filenames = [filename.replace("_ROI0.h5", "") for filename in filenames]

for corename in core_filenames:
    out_pt_file_path = os.path.join(output_folder_pt, corename+'.pt')
    if not os.path.isfile(out_pt_file_path):
        full_path1 = os.path.join(data_folder, corename + "_ROI0.h5")
        full_path2 = os.path.join(data_folder, corename + "_ROI1.h5")


        # Open the first HDF5 file for reading
        try:
            with h5py.File(full_path1, 'r') as file1:
                features1 = file1['features'][:]  # Assuming 'features' is the dataset name
                coords1 = file1['coords'][:]

        # Open the second HDF5 file for reading
            with h5py.File(full_path2, 'r') as file2:
                features2 = file2['features'][:]
                coords2 = file2['coords'][:]

            # merged_features = np.concatenate([features1, features2], axis=0)
            # merged_coords = np.concatenate([coords1, coords2], axis=0)

            # merged_filename = os.path.join(output_folder, corename + '.h5')
            # with h5py.File(merged_filename, 'w') as merged_file:
            #     merged_file.create_dataset('features', data=merged_features)
            #     merged_file.create_dataset('coords', data=merged_coords)

        #pt files
            features1_np = torch.from_numpy(features1)
            features2_np = torch.from_numpy(features2)
            merged_features_np = np.concatenate([features1_np, features2_np], axis=0)
            merged_features_torch = torch.from_numpy(merged_features_np)

            torch.save(merged_features_torch, out_pt_file_path)
        except:
            print("file error", corename)

print('Done: ' + seed_folder)

