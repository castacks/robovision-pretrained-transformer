import torch
import sys
import numpy as np
import tartanair as ta
from torch.utils.data import Dataset, DataLoader

tartanairv2_data_root = 'datasets/tartanairv2' #FIXME
ta.init(tartanairv2_data_root)

def build_train_dataset():

    matching_dataset = ta.create_image_dataset(env = ['ConstructionSite', 'SupermarketExposure'], 
                                                        difficulty = [], trajectory_id = [], 
                                                        modality = ['image', 'depth'], 
                                                        camera_name = ['lcam_front', 'lcam_back', 'lcam_right', 'lcam_left', 'lcam_top', 'lcam_bottom'], 
                                                        transform = None, num_workers=10) #FIXME
    return matching_dataset


# # uncompress the data
# def flow16to32(flow16):
#     '''
#     flow_32b (float32) [-512.0, 511.984375]
#     flow_16b (uint16) [0 - 65535]
#     flow_32b = (flow16 -32768) / 64
#     '''
#     flow32 = flow16[:,:,:2].astype(np.float32)
#     flow32 = (flow32 - 32768) / 64.0

#     mask8 = flow16[:,:,2].astype(np.uint8)
#     return flow32, mask8

