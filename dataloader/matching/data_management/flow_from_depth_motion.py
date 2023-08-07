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
from image_resampling.mvs_utils.camera_models import LinearSphere, ShapeStruct
from visualization.visualization_utils import DataVisualizer


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


if __name__ == '__main__':
    # Config.

    # The directory of the fisheye rgb and depth images.
    fish_dist_images_dir = './example_tartanair/ArchVizTinyHouseDayExposure/Data_hard/P000/depth_lcam_fish'
    fish_dist0_gfps = [os.path.join(fish_dist_images_dir, f) for f in os.listdir(fish_dist_images_dir) if f.endswith('.png')]
    fish_dist0_gfps.sort()

    fish_bgr_images_dir = './example_tartanair/ArchVizTinyHouseDayExposure/Data_hard/P000/image_lcam_fish'
    fish_bgr0_gfps = [os.path.join(fish_bgr_images_dir, f) for f in os.listdir(fish_bgr_images_dir) if f.endswith('.png')]
    fish_bgr0_gfps.sort()

    # The directory of the pinhole rgb and depth images.
    pin_depth_images_dir = './example_tartanair/ArchVizTinyHouseDayExposure/Data_hard/P000/depth_lcam_front'
    pin_depth0_gfps = [os.path.join(pin_depth_images_dir, f) for f in os.listdir(pin_depth_images_dir) if f.endswith('.png')]
    pin_depth0_gfps.sort()

    pin_bgr_images_dir = './example_tartanair/ArchVizTinyHouseDayExposure/Data_hard/P000/image_lcam_front'
    pin_bgr0_gfps = [os.path.join(pin_bgr_images_dir, f) for f in os.listdir(pin_bgr_images_dir) if f.endswith('.png')]
    pin_bgr0_gfps.sort()

    # The directory of the motion vectors.
    pose_gfp = './example_tartanair/ArchVizTinyHouseDayExposure/Data_hard/P000/pose_lcam_fish.txt'


    # The directory of the flow images.
    flow_images_dir = '/home/yorai/Downloads/flow_images'

    # Read the fisheye intrinsics.
    fish_h = 1000
    fish_w = 1000   
    fish_fov_degree = 195 
    fish_shape_struct = ShapeStruct(fish_h, fish_w)
    fish_camera_model = LinearSphere(\
                                    fov_degree=fish_fov_degree, 
                                    shape_struct=fish_shape_struct, out_to_numpy=False)
    valid_mask = fish_camera_model.get_valid_mask()

    pin_camera_model = Pinhole(fx = 320, fy = 320, cx = 320, cy = 320, shape_struct=ShapeStruct(640, 640), out_to_numpy=False)
    pin_camera_model.device = 'cuda'
    # Get the depth images.
    # depth_images = sorted(os.listdir(depth_images

    visualizer = DataVisualizer()
    gt_poses = np.loadtxt(pose_gfp)
    for ix in range(len(fish_dist0_gfps)):
        
        ############################
        # START: A bunch of transformations for no good reason.
        ############################
        # Get the motion. And convert to xyz, rotvec. 
        pos_quat = np.array(gt_poses[ix:ix+2, :])
        traj_motions  = pos_quats2ses(pos_quat) # From xyz, xyzw format, to relative motion (1x6) format.

        sample_motion = traj_motions[0, :]
        X_NED_cam = np.eye(4)
        X_NED_cam[:3, :3] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        X_cam_NED = np.linalg.inv(X_NED_cam)
        X_cube_fish = np.eye(4)

        # Get the motion of the first image. Sample motion is currently of shape (S, 6). With S being the number of samples in a window.
        X_b0_b1 = se2SE(sample_motion)

        # The transform between the base (b) and the rotated base (bb).
        X_b_bb = X_NED_cam @ X_cube_fish @ X_cam_NED

        # The transform between the rotated base (bb) and the base (b).
        X_bb_b = np.linalg.inv(X_b_bb)

        # The NED transformation between the rotated bases.
        X_bb0_bb1 = X_bb_b @ X_b0_b1 @ X_b_bb

        # Back to xyz, rotvec.
        fish_ned_sample_motion = SE2se(X_bb0_bb1)

        # The motion is the same for both fisheye and pinhole.
        ned_gt_motion = fish_ned_sample_motion
        ############################
        # END: A bunch of transformations for no good reason.
        ############################

        ############################
        # Fisheye Flow.
        ############################
        # Load the distance image.
        fish_dist0_gfp = fish_dist0_gfps[ix]
        fish_dist0 = TartanAirImageReader().read_depth(fish_dist0_gfp)
        fish_dist0_img = visualizer.visdepth(fish_dist0)
        fish_dist0 = torch.tensor(fish_dist0, device = 'cuda').float()

        # Compute the optical flow.
        flow_fish_sample = flow_from_dist_motion(fish_dist0, ned_gt_motion, fish_camera_model, device = fish_dist0.device ) # Output is torch.tensor (2, H, W).

        # Load the rgb image.
        fish_bgr0_gfp = fish_bgr0_gfps[ix]
        fish_bgr0 = TartanAirImageReader().read_bgr(fish_bgr0_gfp)

        fish_bgr1_gfp = fish_bgr0_gfps[ix+1]
        fish_bgr1 = TartanAirImageReader().read_bgr(fish_bgr1_gfp)

        # Show toggle flow visualization.
        DataVisualizer().visflow_toggle(flow_fish_sample.cpu().numpy().transpose((1, 2, 0)), fish_bgr0, fish_bgr1, num_toggles=3)

        ############################
        # Pinhole Flow.
        ############################
        # Load the depth image.
        pin_dist0_gfp = pin_depth0_gfps[ix]
        depth0 = TartanAirImageReader().read_depth(pin_dist0_gfp)

        depth0 = torch.tensor(depth0, device = 'cuda').float()
        # Convert to distance. This assumes that the camera model is ideal (90 degrees fov, square image).
        pin_dist0 = depth_to_dist(depth0).cpu().numpy()

        pin_dist0_img = visualizer.visdepth(pin_dist0)
        pin_dist0 = torch.tensor(pin_dist0, device = 'cuda').float()

        # Compute the optical flow.
        flow_pin_sample = flow_from_dist_motion(pin_dist0, ned_gt_motion, pin_camera_model, device = pin_dist0.device ) # Output is torch.tensor (2, H, W).

        # Load the rgb image.
        pin_bgr0_gfp = pin_bgr0_gfps[ix]
        pin_bgr0 = TartanAirImageReader().read_bgr(pin_bgr0_gfp)

        pin_bgr1_gfp = pin_bgr0_gfps[ix+1]
        pin_bgr1 = TartanAirImageReader().read_bgr(pin_bgr1_gfp)

        # Show toggle flow visualization.
        DataVisualizer().visflow_toggle(flow_pin_sample.cpu().numpy().transpose((1, 2, 0)), pin_bgr0, pin_bgr1, num_toggles=3)
