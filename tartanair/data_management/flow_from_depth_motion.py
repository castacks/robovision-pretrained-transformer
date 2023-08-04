'''
Author: Yorai Shaoul
Date: January 2023

This script postprocesses a trajectory of fisheye depth images, alongside their corresponding motion vectors, to produce flow images.
The flow images are saved in the same directory as the depth images, with the same name, but with the extension '.npy' instead of '.png'.
The flow images are saved in the Middlebury format.
'''

# General imports.
import os
import sys
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation

# Local imports.
sys.path.append('..')
from data_management.tartanair_image_reader import TartanAirImageReader
from data_management.transformation import SEs2ses, pose2motion, se2SE, SE2se, pos_quats2SEs,  pose2motion, SEs2ses


def flow_from_depth_motion(dist0, motion01, fish_camera_model, device = 'cuda'):
    '''
    TODO(yoraish): change the input motion to be in the camera frame. Currently it is in an associated NED frame and it is confusing.
    
    dist0: the distance image. A numpy array or torch tensor of shape (H, W). The depth is in meters.
    motion: The motion vector of the fisheye depth image. xyz, rotvec. Note that the motion is of the NED base frame. This is x forward, y right, z down.
    camera_model: The camera model of the fisheye distance image.
    flow_image_gfp: The path to the flow image to be saved.
    '''
    # The motion should be np.array.
    if type(motion01) == torch.Tensor:
        motion01 = motion01.cpu().numpy()

    # Generate pose0 and pose1 from the motion.
    pose0 = np.zeros((7,))
    pose0[-1] = 1.0
    pose1 = np.zeros((7,))
    pose1[0:3] = motion01[0:3]
    pose1[3:7] = Rotation.from_rotvec(motion01[3:6]).as_quat()

    pose0_torch = torch.from_numpy(pose0).float().to(device)
    pose1_torch = torch.from_numpy(pose1).float().to(device)

    # To torch. Definitions:
    # pose0: The pose of the fisheye camera at time 0. In some world frame. This corresponds to the fisheye depth image. [x, y, z, qw, qx, qy, qz].
    # pose1: The pose of the fisheye camera at time 1. This is in [x, y, z, qw, qx, qy, qz].
    if type(dist0) == np.ndarray:
        dist0 = torch.from_numpy(dist0).float().to(device)
    else:
        dist0 = dist0.float().to(device)

    # Get a grid of pixel coordinates in the image.
    G = fish_camera_model.pixel_coordinates(shift = 0.5, flatten = True) # G is a 2xH*W tensor.
    G = G.to(device)

    # Get the rays.
    rays, rays_valid_mask = fish_camera_model.pixel_2_ray(G) # rays is a 3xH*W tensor. 
    rays_valid_mask = rays_valid_mask.view((1, -1))

    rays = rays.to(device)
    rays_valid_mask = rays_valid_mask.to(device)
    
    # Get the depth. Organize it in a 1xH*W tensor.
    dist0 = dist0.view((1, -1))

    # Get the points in the camera frame.
    points0 = rays * dist0 # points is a 3xH*W tensor. The points are in the camera frame (z-axis pointing forward, x-axis pointing right, y-axis pointing down)

    # Get the points in the new image frame.
    R0Inv = torch.tensor(Rotation.from_quat(pose0[3:]).as_matrix(), device = device).float()
    R0 = R0Inv.T

    t0 = -R0 @ (pose0_torch[:3].reshape((-1, 1)))
    R1Inv = torch.tensor(Rotation.from_quat(pose1[3:]).as_matrix()).float()

    R1 = R1Inv.T.float().to(device)
    t1 = -R1 @ (pose1_torch[:3].reshape((-1, 1)))

    # Calculate the coordinates in the first camera's frame.
    X0 = points0.roll(1, 0) # Coordinates in the NED frame. z-axis pointing downwards.

    # The coordinates in the world frame.
    XWorld_0  = R0Inv @ (X0 - t0)

    # The coordinate of the pixels of the first camera projected in the second camera's frame (NED).
    X1 = R1 @ (XWorld_0) + t1

    # Back to xyz from zxy.
    points1 = X1.roll(2, 0) 

    # Get the pixel locations that the new points fall onto.
    G1, valid_mask1 = fish_camera_model.point_3d_2_pixel(points1) # G1 is a 2xH*W tensor.

    # Get the du and dv.
    du = G1[0, :] - G[0, :]
    dv = G1[1,:] - G[1,:]

    # Apply the mask.
    du = du * rays_valid_mask
    dv = dv * rays_valid_mask

    H, W = fish_camera_model.ss.H, fish_camera_model.ss.W

    # Reshape.
    du = du.view((H, W))
    dv = dv.view((H, W))
    
    # The flow image.
    flow = torch.stack((du, dv), dim = 0)
    return flow