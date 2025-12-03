import os
import h5py
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-orig_data_folder', dest='orig_data_folder', type=str, default=None)
parser.add_argument('-aug_data_folder', dest='aug_data_folder', type=str, default=None)
parser.add_argument('-output_folder', dest='output_folder', type=str, default=None)
args = parser.parse_args()

# feature_folder = '/rsrch6/home/trans_mol_path/yuan_lab/TIER2/barrett/clam/features_ROI'
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_files_in_folder(folder):
    # List files in the specified folder
    return [os.path.splitext(f)[0] for f in os.listdir(folder)]


def compare_folders(folder1, folder2):
    # List files in both folders
    files1 = list_files_in_folder(folder1)
    files2 = list_files_in_folder(folder2)

    if set(files1) == set(files2):
        print("Both folders have the same files.")
    else:
        print("Folders have differences.")
        # Files in folder1 but not in folder2
        diff1 = [file for file in files1 if file not in files2]
        # Files in folder2 but not in folder1
        diff2 = [file for file in files2 if file not in files1]

        print("Files in {} but not in {}:".format(folder1, folder2))
        for file in diff1:
            print(file)
        print("\nFiles in {} but not in {}:".format(folder2, folder1))
        for file in diff2:
            print(file)


# folder_list =["h5", "pt"]
# for subfolder in folder_list:

#orig_data_folder = os.path.join(args.feature_folder,"features_roi_orig","merged",subfolder+"_files")
orig_h5_folder = os.path.join(args.orig_data_folder,"h5_files")
#aug_data_folder = os.path.join(args.feature_folder,"features_roi_aug","merged",subfolder+"_files")
aug_h5_folder = os.path.join(args.aug_data_folder,"h5_files")
#output_folder = os.path.join(args.feature_folder,"merged",subfolder+"_files")
output_folder = args.output_folder
print("output folder :", output_folder)
os.makedirs(os.path.join(output_folder, "h5_files"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "pt_files"), exist_ok=True)



orig_filenames = list_files_in_folder(orig_h5_folder)
aug_filenames = list_files_in_folder(aug_h5_folder)

if set(orig_filenames) != set(aug_filenames):
    print("Folders have differences.")
    # Files in folder1 but not in folder2
    diff1 = [file for file in orig_filenames if file not in aug_filenames]
    # Files in folder2 but not in folder1
    diff2 = [file for file in aug_filenames if file not in orig_filenames]

    print("Files in {} but not in {}:".format(orig_filenames, aug_filenames))
    for file in diff1:
        print(file)
    print("\nFiles in {} but not in {}:".format(aug_filenames, orig_filenames))
    for file in diff2:
        print(file)
    



for base_filename in orig_filenames:
    orig_h5_full_path = os.path.join(orig_h5_folder, base_filename +'.h5')
    aug_h5_full_path = os.path.join(aug_h5_folder, base_filename +'.h5')

#if subfolder == "h5":
    # Open the first HDF5 file for reading
    with h5py.File(orig_h5_full_path, 'r') as orig_file:
        orig_features = orig_file['features'][:]  
        #orig_coords = orig_file['coords'][:]

    # Open the second HDF5 file for reading
    with h5py.File(aug_h5_full_path, 'r') as aug_file:
        aug_features = aug_file['features'][:]
        #aug_coords = aug_file['coords'][:]

    merged_features = np.concatenate([orig_features, aug_features], axis=0)
    #merged_coords = np.concatenate([orig_coords, aug_coords], axis=0)

    merged_h5_filename = os.path.join(output_folder, "h5_files", base_filename +'.h5')
    with h5py.File(merged_h5_filename, 'w') as merged_file:
        merged_file.create_dataset('features', data=merged_features)
        #merged_file.create_dataset('coords', data=merged_coords)

# if subfolder == "pt":
#     # Load labels/metadata from the first PyTorch file
#     labels1 = torch.load(orig_full_path)

#     # Load labels/metadata from the second PyTorch file
#     labels2 = torch.load(aug_full_path)

#     # Merge the labels/metadata (adjust the merging logic based on your data)
#     merged_labels = torch.cat([labels1, labels2], dim=0)
#     merged_filename = os.path.join(output_folder, filename)
#     # Save the merged labels/metadata to a new PyTorch file
#     torch.save(merged_labels, merged_filename)
    merged_features = torch.from_numpy(merged_features)
    merged_pt_filename = os.path.join(output_folder, "pt_files", base_filename +'.pt')
    torch.save(merged_features, merged_pt_filename)

