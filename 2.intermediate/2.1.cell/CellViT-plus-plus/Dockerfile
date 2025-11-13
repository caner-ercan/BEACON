FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y \
    wget curl git build-essential ca-certificates sudo \
    && rm -rf /var/lib/apt/lists/*

RUN rm -rf /opt/conda
RUN rm -rf ~/.condarc ~/.conda ~/.continuum
ENV CONDA_DIR=/opt/conda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh && \
    conda clean -afy

RUN conda create -n cellvit python=3.10.14 -y  --channel nvidia --channel rapidsai --channel conda-forge
SHELL ["conda", "run", "-n", "cellvit", "/bin/bash", "-c"]
RUN conda install rapidsai::cucim=24.04.00=cuda12_py310_240410_ga24abfd_0 cupy=13.3.0=py310h1b77274_0 cupy-core=13.3.0=py310hc8bf42c_0 numpy=1.23.5=py310h53a5b5f_0 \
openslide=4.0.0=h58ba908_1 pillow=10.3.0=py310hf73ecf8_0

# RUN conda config -–remove channels defaults
WORKDIR /tmp
RUN wget https://raw.githubusercontent.com/TIO-IKIM/CellViT-plus-plus/main/requirements.txt -O requirements.txt

RUN pip install -r requirements.txt
RUN conda install -y ipykernel && \
    python -m ipykernel install --prefix=/opt/conda --name cellvit --display-name 'Python310_cellvit'
ENV CONDA_DEFAULT_ENV=cellvit
ENV CONDA_PREFIX=/opt/conda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN source /opt/conda/etc/profile.d/conda.sh


WORKDIR /workspace
