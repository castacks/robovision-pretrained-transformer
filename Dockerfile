FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

#RUN useradd -m mihirsharma

#RUN chown -R mihirsharma:mihirsharma /home/mihirsharma/

# RUN useradd -m -s /bin/bash mihirsharma

# USER mihirsharma

COPY . /app

WORKDIR /app

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && apt-get -y install sudo \
    && rm -rf /var/lib/apt/lists/*

RUN sudo apt install python3

RUN apt-get install -y python3 python3-pip

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN export PATH="$HOME/opt/git/bin:$PATH"

RUN conda env create -f conda_environment.yml



# RUN CHECKPOINT_DIR=checkpoints_flow/chairs-gmflow-scale1 && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_matching.py \
# --launcher pytorch \
# --checkpoint_dir ${CHECKPOINT_DIR} \
# --stage sintel \
# --batch_size 2 \
# --lr 4e-4 \
# --image_size 640 640 \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log