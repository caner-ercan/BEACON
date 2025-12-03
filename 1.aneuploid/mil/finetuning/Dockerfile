FROM tensorflow/tensorflow:2.12.0-gpu
# RUN apt-get install -y python3-dev

RUN pip install tensorflow-similarity

# RUN pip install protobuf==3.20.3
RUN pip install tensorflow-addons
RUN pip install tensorflow-hub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 openslide-tools 
RUN pip install openslide-bin opencv-python openslide-python==1.3.1 albumentations

WORKDIR /rsrch5/home/trans_mol_path/cercan/code/Barretts/clam_midl_code/aneuploid_repo/