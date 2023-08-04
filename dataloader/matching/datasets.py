# Data loading based on:
# https://github.com/NVIDIA/flownet2-pytorch
# https://github.com/princeton-vl/RAFT
# https://github.com/haofeixu/gmflow

import numpy as np
import torch
import torch.utils.data as data

import os
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from dataloader.flow.transforms import FlowAugmentor, SparseFlowAugmentor


class TartanAirV2(data.Dataset):
    placeholder=1#FIXME
