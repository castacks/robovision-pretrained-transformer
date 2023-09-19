FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

#RUN useradd -m mihirsharma

#RUN chown -R mihirsharma:mihirsharma /home/mihirsharma/


RUN useradd -m -s /bin/bash mihirsharma

USER mihirsharma

COPY . /app

WORKDIR /app

RUN sudo apt-get update && sudo apt install python3.

RUN pip install conda

RUN cd robovision-pretrained-transformer && conda env create -f conda_environment.yml

RUN CHECKPOINT_DIR=checkpoints_flow/chairs-gmflow-scale1 && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main_matching.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--stage sintel \
--batch_size 2 \
--lr 4e-4 \
--image_size 640 640 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log
