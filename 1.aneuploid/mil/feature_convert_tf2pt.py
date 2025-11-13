import os
import h5py
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', dest='feature_folder', type=str, default=None)
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_folder = os.path.join(args.feature_folder,"h5_files")
output_folder_pt = os.path.join(args.feature_folder,"pt_files")
print("writing features to", output_folder_pt)
os.makedirs(output_folder_pt, exist_ok=True)
filenames = os.listdir(data_folder)

for fname in filenames:
    input_file = os.path.join(data_folder,fname)
    output_file = os.path.join(output_folder_pt,os.path.splitext(fname)[0]+".pt")

    with h5py.File(input_file, 'r') as file:
            features = file['features'][:]  

    features_torch = torch.from_numpy(features)        

    torch.save(features_torch, output_file)

print('Done: '+ args.feature_folder)
