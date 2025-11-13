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
folder_list =["h5", "pt"]
for subfolder in folder_list:

    data_folder = os.path.join(seed_folder,subfolder+"_files")
    output_folder = os.path.join(seed_folder,"merged",subfolder+"_files")
    print("writing features to", output_folder)
    os.makedirs(output_folder, exist_ok=True)
    filenames = [filename for filename in os.listdir(data_folder) if filename.endswith("ROI0."+subfolder)]

    core_filenames = [filename.replace("_ROI0."+subfolder, "") for filename in filenames]

    for corename in core_filenames:
        full_path1 = os.path.join(data_folder, corename + "_ROI0."+subfolder)
        full_path2 = os.path.join(data_folder, corename + "_ROI1."+subfolder)

        if subfolder == "h5":
            # Open the first HDF5 file for reading
            with h5py.File(full_path1, 'r') as file1:
                features1 = file1['features'][:]  # Assuming 'features' is the dataset name

            # Open the second HDF5 file for reading
            with h5py.File(full_path2, 'r') as file2:
                features2 = file2['features'][:]

            merged_features = np.concatenate([features1, features2], axis=0)

            merged_filename = os.path.join(output_folder, corename + '.h5')
            with h5py.File(merged_filename, 'w') as merged_file:
                merged_file.create_dataset('features', data=merged_features)

        if subfolder == "pt":
            # Load labels/metadata from the first PyTorch file
            labels1 = torch.load(full_path1)

            # Load labels/metadata from the second PyTorch file
            labels2 = torch.load(full_path2)

            # Merge the labels/metadata (adjust the merging logic based on your data)
            merged_labels = torch.cat([labels1, labels2], dim=0)
            merged_filename = os.path.join(output_folder, corename + '.pt')
            # Save the merged labels/metadata to a new PyTorch file
            torch.save(merged_labels, merged_filename)