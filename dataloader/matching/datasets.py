import torch
import sys
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