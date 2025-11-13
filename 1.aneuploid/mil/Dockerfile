from pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
# set environment variables
ENV DEBIAN_FRONTEND noninteractive
ENV HDF5_USE_FILE_LOCKING FALSE
ENV NUMBA_CACHE_DIR /tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential libgl1-mesa-glx libglib2.0-0 libgeos-dev libvips-tools libffi-dev \
  sudo curl wget htop git vim ca-certificates  \
  && rm -rf /var/lib/apt/lists/*

RUN conda install -y tensorflow=2
RUN conda install -y openslide=3.4.1 -c conda-forge

WORKDIR /app
RUN git clone https://github.com/oval-group/smooth-topk.git
WORKDIR /app/smooth-topk
RUN python setup.py install

ENV TORCH_HOME /app/torch/
WORKDIR /app/torch/hub/checkpoints
RUN wget https://download.pytorch.org/models/resnet50-19c8e357.pth
RUN wget https://download.pytorch.org/models/densenet121-a639ec97.pth
RUN chmod -R 777 /app/torch/hub/checkpoints

RUN python -m pip install --upgrade pip
RUN pip install opencv-python openslide-python==1.3.1 scikit-learn matplotlib pandas albumentations tensorboardX numpy==1.23.5

WORKDIR /.local
RUN chmod -R 777 /.local
WORKDIR /.cache
RUN chmod -R 777 /.cache


# bash at /App
WORKDIR /workspace
CMD ["/bin/bash"]