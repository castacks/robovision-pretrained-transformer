FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

COPY . /app

WORKDIR /app

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && apt-get -y install sudo \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y python3.9 python3.9-dev

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda install git

RUN export PATH="$HOME/opt/git/bin:$PATH"

RUN conda env create -f conda_environment.yml
