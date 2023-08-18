import sys
import os
import cv2
import numpy as np
import torch
import torchvision
import random
import math
###########################################################################
# import tartanair
###########################################################################
LOCAL_PI = math.pi


def flow16to32(flow16):
    '''
    flow_32b (float32) [-512.0, 511.984375]
    flow_16b (uint16) [0 - 65535]
    flow_32b = (flow16 -32768) / 64
    '''
    flow32 = flow16[:, :, :2].astype(np.float32)
    flow32 = (flow32 - 32768) / 64.0

    mask8 = flow16[:, :, 2].astype(np.uint8)
    return flow32, mask8

sys.path.append('..')

###########################################################################
# from pinhole_flow import process_single_process, CameraBase
# The directory of the pinhole rgb images.
# pin_bgr_images_dir = '/home/mihirsharma/AirLab/datasets/tartanair/image_lcam_front'
# pin_bgr0_gfps = [os.path.join(pin_bgr_images_dir, f) for f in os.listdir(
#     pin_bgr_images_dir) if f.endswith('.png')]
# pin_bgr0_gfps.sort()

# pin_flow_images_dir = '/home/mihirsharma/AirLab/datasets/tartanair/flow_lcam_front'
# pin_flow_gfps = [os.path.join(pin_flow_images_dir, f) for f in os.listdir(
#     pin_flow_images_dir) if f.endswith('.png')]
# pin_flow_gfps.sort()
######################################################################################################################################################
# mask_list = []
# tartanair.init('/home/mihirsharma/AirLab/datasets/tartanair')
# tartan_air_dataloader = tartanair.dataloader(env = 'AbandonedFactoryExposure', difficulty = 'easy', trajectory_id = ['P000'], modality = ['image', 'flow'], camera_name = 'lcam_front', batch_size = 4)
######################################################################################################################################################

def convert_flow_batch_to_matching(batch, crop_size=[1/4, 1/4], downsample_size=8, standard_deviation=1.5, samples = 400, device = 'cuda'):

    images_tensor = batch['rgb_lcam_front'].squeeze(dim=1).permute(0, 3, 1, 2).to(device) #[B+1, 3, H, W] g
    
    #Tensor sizing variables
    b, c, h, w = images_tensor.shape
    batch_size = b - 1
    feature_map_height = round(h / downsample_size)
    feature_map_width = round(w / downsample_size)
    feature_map_crop_height = round(h * crop_size[0] / downsample_size)
    feature_map_crop_width = round(w * crop_size[1] / downsample_size)
    

    images_1_tensor = images_tensor[:-1] #[B, 2, H, W] g
    images_2_tensor = images_tensor[1:] #[B, 2, H, W] g


    ######################################################################################################################################################
    # print(images_1_tensor.shape)
    # print(images_2_tensor.shape)
    # cv2.imwrite('test_images/image1.png', (images_1_tensor[0]).permute(1, 2, 0).to('cpu').numpy().astype(np.uint8))
    # cv2.imwrite('test_images/image2.png', (images_2_tensor[0]).permute(1, 2, 0).to('cpu').numpy().astype(np.uint8))
    # images_tensor_1 = torch.strided_view(images_tensor, )
    ######################################################################################################################################################
    

    # Random token selection
    random_samples_reference_return = torch.randint(high = feature_map_height * feature_map_width, size=(batch_size, samples)).unsqueeze(dim=1).to(device) #[B, 1, Samples] g
    random_samples_reference = random_samples_reference_return.repeat(1, 2, 1) #[B, 2, Samples] g

    
    # add mask
    #Mask gets set to 1 for valid and 0 for invalid
    mask = batch['flow_lcam_front'].squeeze(dim=1).permute(0, 3, 1, 2).to(device)[:-1][:, 2, :, :].unsqueeze(dim=1) #[B, 1, 640, 640]
    # Masked pixels become 0
    mask = torch.where(mask != 0.0, 0.0, 1.0) #good
    # cv2.imwrite('test_images/mask.png', (mask[0].permute(1, 2, 0) * 255).to('cpu').numpy().astype(np.uint8))


    flow = batch['flow_lcam_front'].squeeze(dim=1).permute(0, 3, 1, 2).to(device)[:-1][:, :-1, :, :]  # [B, 2, H, W] g;
    masked_flow = flow * mask #[B, 2, H, W]
    feature_flow = torch.div(torch.nn.functional.avg_pool2d(masked_flow, kernel_size=downsample_size, stride=downsample_size), downsample_size)  # [B, 2, Feature H, Feature W] g
    b_ff, c_ff, h_ff, w_ff = feature_flow.shape
    flattened_feature_flow = feature_flow.contiguous().view(b_ff, c_ff, h_ff * w_ff) #[B, 2, Feature Area] g
    selective_flow = torch.gather(input=flattened_feature_flow, dim=2, index=random_samples_reference) #[B, 2, Samples] g


    x_coords = ((random_samples_reference[:, 0, :]) % w_ff).unsqueeze(dim=-1).permute(2, 0, 1) #[1, B, Samples] g
    y_coords = torch.floor(torch.div(random_samples_reference[:, 0, :], w_ff)).unsqueeze(dim=-1).permute(2, 0, 1).to(torch.int16) #[1, B, Samples] g
    x_y_coords = torch.cat([x_coords, y_coords]).permute(1, 0, 2).to(torch.int16) #[B, 2, Samples] g


    correlation_positions_x_y = torch.add(selective_flow, x_y_coords).permute(0, 2, 1) #[B, Samples, 2] g


    batch_size, correlation_samples, correlation_dimensions = correlation_positions_x_y.shape


    x_meshgrid, y_meshgrid = torch.meshgrid(torch.arange(0, feature_map_width), torch.arange(0, feature_map_height)) #[Feature H, Feature W] x 2 g
    x_meshgrid = x_meshgrid.to(device).unsqueeze(dim = 0).repeat(batch_size, 1, 1) #[B, Feature H, Feature W] g
    y_meshgrid = y_meshgrid.to(device).unsqueeze(dim = 0).repeat(batch_size, 1, 1) #[B, Feature H, Feature W] g
    x_meshgrid = x_meshgrid.unsqueeze(1).repeat(1, samples, 1, 1) #[B, Samples, Feature H, Feature W] g
    y_meshgrid = y_meshgrid.unsqueeze(1).repeat(1, samples, 1, 1) #[B, Samples, Feature H, Feature W] g


    x_mc = x_meshgrid - correlation_positions_x_y[..., 0].contiguous().view(batch_size, samples, 1, 1) #[B, Samples, Feature W, Feature H] g
    y_mc = y_meshgrid - correlation_positions_x_y[..., 1].contiguous().view(batch_size, samples, 1, 1) #[B, Samples, Feature W, Feature H] g
   

    squared_distance = x_mc ** 2 + y_mc ** 2 #[B, Samples, Feature W, Feature H] g
    gaussian = torch.exp((-1 * squared_distance) / (standard_deviation ** 2)) #[B, Samples, Feature W, Feature H] g
    gaussian = gaussian.permute(0, 1, 3, 2) #[B, Samples, Feature H, Feature W] g


    cropped_gaussian = torch.zeros(size = (batch_size, samples, feature_map_crop_height, feature_map_crop_width)).to(device) #[B, Samples, Crop H, Crop W] g


    random_crop_locations_subtract_x_y = torch.randint(high = 12, size = (batch_size, 2, samples)).to(device) #[B, 2, Samples] g
    random_crop_locations_x_y = (correlation_positions_x_y.permute(0, 2, 1) - random_crop_locations_subtract_x_y).to(torch.int16) #[B, 2, Samples] g
    random_crop_locations_x_y = torch.where(random_crop_locations_x_y > feature_map_width-1, feature_map_width-1, random_crop_locations_x_y) #[B, 2, Samples] g
    random_crop_locations_x_y = torch.where(random_crop_locations_x_y < 0, 0, random_crop_locations_x_y) #[B, 2, Samples] g

    

    for i in range(gaussian.shape[0]):
        for j in range(gaussian.shape[1]):
            cropped_gaussian[i, j] = torchvision.transforms.functional.crop(img = gaussian[i, j], top = random_crop_locations_x_y[i, 1, j].to('cpu').numpy(), left = random_crop_locations_x_y[i, 0, j].to('cpu').numpy(), height = feature_map_crop_height, width = feature_map_crop_width)
    
    
    ######################################################################################################################################################
    # print(correlation_positions_x_y[0, 199, :])
    # print(random_crop_locations_x_y[0, :, 199])
    # cv2.imwrite('test_images/rgb-gaussian.png', (gaussian[0, 199, :, :] * 255).unsqueeze(dim=0).permute(1, 2, 0).to('cpu').numpy().astype(np.uint8))
    # cv2.imwrite('test_images/rgb-cropped_gaussian.png', (cropped_gaussian[0, 199, :, :] * 255).unsqueeze(dim=0).permute(1, 2, 0).to('cpu').numpy().astype(np.uint8))
    ######################################################################################################################################################


    feature_map_crop_shape = [feature_map_crop_height, feature_map_crop_width]
    return images_1_tensor, images_2_tensor, cropped_gaussian, random_samples_reference_return, random_crop_locations_x_y, feature_map_crop_shape, samples
  

    
######################################################################################################################################################
# convert_flow_batch_to_matching(tartan_air_dataloader.load_sample(), crop_size=[1/4, 1/4], downsample_size=8, standard_deviation=1.25, device = 'cuda')
######################################################################################################################################################