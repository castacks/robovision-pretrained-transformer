import sys
import os
import cv2
import numpy as np
import torch
import torchvision
import random
import math

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
mask_list = []

# tartanair.init('/home/mihirsharma/AirLab/datasets/tartanair')

# tartan_air_dataloader = tartanair.dataloader(env = 'AbandonedFactoryExposure', difficulty = 'easy', trajectory_id = ['P000'], modality = ['image', 'flow'], camera_name = 'lcam_front', batch_size = 12)

# tartan_air_batch = tartan_air_dataloader.load_sample()


def convert_flow_batch_to_matching(batch, crop_size=[1/4, 1/4], downsample_size=8, standard_deviation=2, device = 'cuda'):

    # cv2.imwrite('images_tensor.png', images_tensor.to('cpu').numpy().astype(np.uint8))

    images_tensor = batch['rgb_lcam_front'].squeeze(dim=1).permute(0, 3, 1, 2) #[Batch(12), Channels(3), Height(640), Width(640)]

    images_1_tensor = images_tensor[::2]

    images_2_tensor = images_tensor[1::2]

    # print(images_1_tensor.shape)
    # print(images_2_tensor.shape)

    # cv2.imwrite('test_images/image1.png', (images_1_tensor[0]).permute(1, 2, 0).to('cpu').numpy().astype(np.uint8))

    # cv2.imwrite('test_images/image2.png', (images_2_tensor[0]).permute(1, 2, 0).to('cpu').numpy().astype(np.uint8))

    # images_tensor_1 = torch.strided_view(images_tensor, )
    #Tensor sizing variables
    b, c, h, w = images_tensor.shape
    batch_size = b
    feature_map_height = round(h / downsample_size)
    feature_map_width = round(w / downsample_size)
    feature_map_crop_height = round(h * crop_size[0] / downsample_size)
    feature_map_crop_width = round(w * crop_size[1] / downsample_size)
    

    # Random token selection
    random_samples_reference = torch.randint(high = feature_map_height * feature_map_width, size=(batch_size, feature_map_crop_height * feature_map_crop_width)).unsqueeze(dim=1).repeat(1, 2, 1).to('cuda')  # [Batch(12), Crop Area(400)]
    # print(random_samples_reference.shape)
    flow = batch['flow_lcam_front'].squeeze(dim=1).permute(0, 3, 1, 2).to('cuda')  # [Batch, H(640), W(640), 2]

    #TODO add mask

    # #Mask gets set to 1 for valid and 0 for invalid
    # mask = torch.where(mask != 0.0, 0.0, 1.0) #good

    # print(mask[0, 500, 500, 0])

    # cv2.imwrite('mask.png', (mask[140] * 255).to('cpu').numpy().astype(np.uint8))

    # # Masked pixels become 0
    # masked_flow = torch.mul(flow, mask).permute(0, 3, 1, 2)  # [Batch, 2, H(640), W(640)], seems good

    feature_flow = torch.div(torch.nn.functional.avg_pool2d(flow, kernel_size=downsample_size, stride=downsample_size), downsample_size)  # [Batch, 2, H/downsample(80), W/downsample(80)]

    b_ff, c_ff, h_ff, w_ff = feature_flow.shape

    # # [2, Batch * H/downsample(80) * W/downsample(80) (6400 * 12)]
    # flattened_feature_flow = feature_flow.permute(1, 0, 2, 3).view(c_ff, b_ff, h_ff * w_ff).view(c_ff, b_ff * h_ff * w_ff)
    
    # # [Batch * feature_map_crop_height * feature_map_crop_width (400 * 12)]
    # selective_masked_flow_indices = torch.floor(random_samples_reference * e_ff * h_ff * w_ff).to(torch.int32)

    # # [2, Batch * feature_map_crop_height * feature_map_crop_width (400 * 12)]
    # selective_masked_flow = torch.index_select(flattened_feature_flow, 1, selective_masked_flow_indices)

    flattened_feature_flow = feature_flow.view(b_ff, c_ff, h_ff * w_ff)
    selective_flow = torch.gather(input=flattened_feature_flow, dim=2, index=random_samples_reference) #CORRECT, GOOD

    x_coords = ((random_samples_reference[:, 0, :]) % w_ff).unsqueeze(dim=-1).permute(2, 0, 1)  # [1, 400]

    y_coords = torch.floor(torch.div(random_samples_reference[:, 0, :], w_ff)).unsqueeze(dim=-1).permute(2, 0, 1)  # [1, feature_map_crop_height * feature_map_crop_width (400)]

    x_y_coords = torch.cat([x_coords, y_coords]).permute(1, 0, 2)

    correlation_positions = torch.add(selective_flow, x_y_coords).permute(0, 2, 1) # good, [Batch, feature_map_crop_height * feature_map_crop_width aka samples (400), 2]

    batch_size, correlation_samples, correlation_dimensions = correlation_positions.shape  # good - 400, 2


    x_meshgrid, y_meshgrid = torch.meshgrid(torch.arange(0, feature_map_width), torch.arange(0, feature_map_height)) #, indexing='xy'  # [Batch, H/downsample(80), W/downsample(80)] x 2

    x_meshgrid = x_meshgrid.unsqueeze(dim = 0).repeat(batch_size, 1, 1)
    y_meshgrid = y_meshgrid.unsqueeze(dim = 0).repeat(batch_size, 1, 1)
    x_meshgrid = x_meshgrid.unsqueeze(1).repeat(1, feature_map_crop_height * feature_map_crop_width, 1, 1).to('cuda') # [feature_map_crop_height * feature_map_crop_width aka samples (400), H/downsample(80), W/downsample(80)] # good

    # [feature_map_crop_height * feature_map_crop_width aka samples (400), H/downsample(80), W/downsample(80)]
    y_meshgrid = y_meshgrid.unsqueeze(1).repeat(1, feature_map_crop_height * feature_map_crop_width, 1, 1).to('cuda') # good 
    
    #[feature_map_crop_height * feature_map_crop_width aka samples (400), H/downsample(80), W/downsample(80)]
    x_mc = x_meshgrid - correlation_positions[..., 0].contiguous().view(batch_size, feature_map_crop_height * feature_map_crop_width, 1, 1) #good

    # [feature_map_crop_height * feature_map_crop_width aka samples (400), H/downsample(80), W/downsample(80)]
    y_mc = y_meshgrid - correlation_positions[..., 1].contiguous().view(batch_size, feature_map_crop_height * feature_map_crop_width, 1, 1) #good

    # [feature_map_crop_height * feature_map_crop_width aka samples (400), H/downsample(80), W/downsample(80)]
    squared_distance = x_mc ** 2 + y_mc ** 2

    # [feature_map_crop_height * feature_map_crop_width aka samples (400), H/downsample(80), W/downsample(80)]
    gaussian = torch.exp((-1 * squared_distance) / (standard_deviation ** 2))

    random_crop_x = random.randint(0, feature_map_width-1)
    random_crop_y = random.randint(0, feature_map_height-1)

    # cropped_gaussian = torch.empty(size = (batch_size, 400, feature_map_crop_height, feature_map_crop_width))


    gaussian = torchvision.transforms.functional.crop(img = gaussian, top = random_crop_y, left = random_crop_x, 
                                                      height = feature_map_crop_height, width = feature_map_crop_width)
    
    random_crop_location = [random_crop_x, random_crop_y]

    feature_map_crop_shape = [feature_map_crop_height, feature_map_crop_width]

    # cv2.imwrite('test_images/rgb-gaussian.png', (gaussian[0, 0, :, :] * 255).unsqueeze(dim=0).permute(1, 2, 0).repeat(1, 1, 3).to('cpu').numpy().astype(np.uint8))

    return images_1_tensor.to('cuda'), images_2_tensor.to('cuda'), gaussian.to('cuda'), x_y_coords.to('cuda'), random_samples_reference.to('cuda'), random_crop_location, feature_map_crop_shape
  

    

# convert_flow_batch_to_matching(tartan_air_batch, crop_size=[1/4, 1/4], downsample_size=8, standard_deviation=2, device = 'cuda')