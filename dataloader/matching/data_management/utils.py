from __future__ import division
import re
import torch
import math
import random
import numpy as np
import numbers
import cv2
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
from scipy.spatial.transform import Rotation

import os
if ( not ( "DISPLAY" in os.environ ) ):
    plt.switch_backend('agg')
    print("Environment variable DISPLAY is not present in the system.")
    print("Switch the backend of matplotlib to agg.")

import time
# ===== general functions =====

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class DownscaleFlow(object):
    """
    Scale the flow and mask to a fixed size
    """
    def __init__(self, scale=4):
        '''
        size: output frame size, this should be NO LARGER than the input frame size! 
        '''
        self.downscale = 1.0/scale

    def __call__(self, sample): 
        if self.downscale!=1 and 'flow' in sample :
            sample['flow'] = cv2.resize(sample['flow'], 
                (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)

        if self.downscale!=1 and 'intrinsic' in sample :
            if len(sample['intrinsic'].shape) == 4:
                # TODO(yoraish): this is a dumb hacky implementation.
                rescaled_intrinsic = np.zeros((sample['intrinsic'].shape[0], \
                                               int(sample['intrinsic'].shape[1] * self.downscale), \
                                               int(sample['intrinsic'].shape[2] * self.downscale), \
                                               sample['intrinsic'].shape[3]))
                for ix in range(sample['intrinsic'].shape[0]):
                    rescaled_intrinsic[ix, :, :, :] = cv2.resize(sample['intrinsic'][ix, :, :, :], 
                        (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)
                sample['intrinsic'] = rescaled_intrinsic
            else:
                sample['intrinsic'] = cv2.resize(sample['intrinsic'], 
                    (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)

        if self.downscale!=1 and 'fmask' in sample :
            sample['fmask'] = cv2.resize(sample['fmask'],
                (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)
        return sample

class CropCenter(object):
    """Crops the a sample of data (tuple) at center
    if the image size is not large enough, it will be first resized with fixed ratio
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        kks = list(sample.keys())
        th, tw = self.size
        h, w = sample[kks[0]].shape[0], sample[kks[0]].shape[1]
        if w == tw and h == th:
            return sample

        # resize the image if the image size is smaller than the target size
        scale_h, scale_w, scale = 1., 1., 1.
        if th > h:
            scale_h = float(th)/h
        if tw > w:
            scale_w = float(tw)/w
        if scale_h>1 or scale_w>1:
            scale = max(scale_h, scale_w)
            w = int(round(w * scale)) # w after resize
            h = int(round(h * scale)) # h after resize

        x1 = int((w-tw)/2)
        y1 = int((h-th)/2)

        for kk in kks:
            if kk == "pinx_in_fish" or kk == "fish_in_pinx":
                continue
            if sample[kk] is None:
                continue
            img = sample[kk]
            if len(img.shape)==3:
                if scale>1:
                    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_LINEAR)
                sample[kk] = img[y1:y1+th,x1:x1+tw,:]
            elif len(img.shape)==2:
                if scale>1:
                    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_LINEAR)
                sample[kk] = img[y1:y1+th,x1:x1+tw]

        return sample


class ToTensor(object):
    def __call__(self, sample):
        sss = time.time()

        kks = list(sample)

        for kk in kks:
            data = sample[kk]

            # Do not operate on inputs that are already tensors. 
            if type(data) == torch.Tensor:
                data.to(dtype = torch.float32)

                if len(data.shape) == 4: # transpose image-like data to get (B, C, H, W).
                    data = data.permute(0,3,1,2)
                if len(data.shape) == 3: # transpose image-like data
                    data = data.permute(2,0,1)
                elif len(data.shape) == 2:
                    data = data.permute((1,)+data.shape)

                if len(data.shape) == 3 and data.shape[0]==3: # normalization of rgb images
                    data = data/255.0

                if len(data.shape) == 4 and data.shape[1]==3: # normalization of rgb images
                    data = data/255.0
                print(data.shape)
                sample[kk] = data.clone()
            else:
                data = data.astype(np.float32)
                if len(data.shape) == 4: # transpose image-like data to get (B, C, H, W).
                    data = data.transpose(0,3,1,2)
                if len(data.shape) == 3: # transpose image-like data
                    data = data.transpose(2,0,1)
                elif len(data.shape) == 2:
                    data = data.reshape((1,)+data.shape)

                if len(data.shape) == 3 and data.shape[0]==3: # normalization of rgb images
                    data = data/255.0

                if len(data.shape) == 4 and data.shape[1]==3: # normalization of rgb images
                    data = data/255.0

                sample[kk] = torch.from_numpy(data.copy()) # copy to make memory continuous

        return sample


class Transform():
    def __init__(self, transform = pt.transform_from_pq([0,0,0, 1,0,0,0]), frame_id = 'world', child_frame_id = 'cam') -> None:
        self.transform = transform
        self.frame_id = frame_id
        self.child_frame_id = child_frame_id

    def __str__(self) -> str:
        out = ""
        out += f"frame_id: {self.frame_id} \n"
        out += f"child_frame_id: {self.child_frame_id}\n"
        T_norm = np.linalg.norm(self.transform[:3, 3])
        if T_norm == 0:
            T_norm = 1
        out += f"Direction T," + str(self.transform[0:3, 3].T/T_norm) + "\n"
        out += f"          T," + str(self.transform[0:3, 3].T) + "\n"
        out+=  f"          R," + str(Rotation.as_euler(Rotation.from_matrix(self.transform[:3, :3]), 'xyz')) + "\n"
        return out


def tensor2img(tensImg,mean,std):
    """
    convert a tensor a numpy array, for visualization
    """
    # undo normalize
    for t, m, s in zip(tensImg, mean, std):
        t.mul_(s).add_(m) 
    tensImg = tensImg * float(255)
    # undo transpose
    tensImg = (tensImg.numpy().transpose(1,2,0)).astype(np.uint8)
    return tensImg

def bilinear_interpolate(img, h, w):
    # assert round(h)>=0 and round(h)<img.shape[0]
    # assert round(w)>=0 and round(w)<img.shape[1]

    h0 = int(math.floor(h))
    h1 = h0 + 1
    w0 = int(math.floor(w))
    w1 = w0 + 1

    a = h - h0 
    b = w - w0

    h0 = max(h0, 0)
    w0 = max(w0, 0)
    h1 = min(h1, img.shape[0]-1)
    w1 = min(w1, img.shape[1]-1)

    A = img[h0,w0,:]
    B = img[h1,w0,:]
    C = img[h0,w1,:]
    D = img[h1,w1,:]

    res = (1-a)*(1-b)*A + a*(1-b)*B + (1-a)*b*C + a*b*D

    return res 

def calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    a = np.arctan2( dv, du )

    angleShift = np.pi

    if ( True == flagDegree ):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")

    d = np.sqrt( du * du + dv * dv )

    return a, d, angleShift

def visflow(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0): 
    """
    Show a optical flow field as the KITTI dataset does.
    Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
    """

    ang, mag, _ = calculate_angle_distance_from_du_dv( flownp[:, :, 0], flownp[:, :, 1], flagDegree=False )

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.float32)

    am = ang < 0
    ang[am] = ang[am] + np.pi * 2

    hsv[ :, :, 0 ] = np.remainder( ( ang + angShift ) / (2*np.pi), 1 )
    hsv[ :, :, 1 ] = mag / maxF * n
    hsv[ :, :, 2 ] = (n - hsv[:, :, 1])/n

    hsv[:, :, 0] = np.clip( hsv[:, :, 0], 0, 1 ) * hueMax
    hsv[:, :, 1:3] = np.clip( hsv[:, :, 1:3], 0, 1 ) * 255
    hsv = hsv.astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if ( mask is not None ):
        mask = mask != 255
        bgr[mask] = np.array([0, 0 ,0], dtype=np.uint8)

    return bgr


def dataset_intrinsics(dataset='tartanair'):
    if dataset == 'kitti':
        focalx, focaly, centerx, centery = 707.0912, 707.0912, 601.8873, 183.1104
    elif dataset == 'euroc':
        focalx, focaly, centerx, centery = 458.6539916992, 457.2959899902, 367.2149963379, 248.3750000000
    elif dataset == 'tartanair-fisheye': # The intrinsics of individual virtual pinhole cameras.
        focalx, focaly, centerx, centery = 600.0, 600.0, 320.0, 224.0    
    elif dataset == 'tartanair':
        focalx, focaly, centerx, centery = 320.0, 320.0, 320.0, 224.0 
    elif dataset == 'ord':
        focalx, focaly, centerx, centery = 320.0, 320.0, 320.0, 224.0
    else:
        return None
    return focalx, focaly, centerx, centery



def plot_traj(gtposes, estposes, vis=False, savefigname=None, title=''):
    fig = plt.figure(figsize=(4,4))
    cm = plt.cm.get_cmap('Spectral')

    plt.subplot(111)
    plt.plot(gtposes[:,0],gtposes[:,1], linestyle='dashed',c='k')
    plt.plot(estposes[:, 0], estposes[:, 1],c='#ff7f0e')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend(['Ground Truth','TartanVO'])
    plt.title(title)
    if savefigname is not None:
        plt.savefig(savefigname)
    if vis:
        plt.show()
    plt.close(fig)

def make_intrinsics_layer(w, h, fx, fy, ox, oy):
    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - ox + 0.5 )/fx
    hh = (hh.astype(np.float32) - oy + 0.5 )/fy
    intrinsicLayer = np.stack((ww,hh)).transpose(1,2,0)

    return intrinsicLayer

def load_kiiti_intrinsics(filename):
    '''
    load intrinsics from kitti intrinsics file
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
    cam_intrinsics = lines[2].strip().split(' ')[1:]
    focalx, focaly, centerx, centery = float(cam_intrinsics[0]), float(cam_intrinsics[5]), float(cam_intrinsics[2]), float(cam_intrinsics[6])

    return focalx, focaly, centerx, centery

def rpy_to_xyzw(r, p, y):
   from scipy.spatial.transform import Rotation 
   return Rotation.from_euler('xyz', [r,p,y], degrees=True).as_quat() # Returns xyzw.


def x_to_norm_xyz_euler_str(x):
    """
    x is a 4x4 transformation matrix.

    """
    out = ""
    out += f"    T GT," + str(x[0:3, 3].T/np.linalg.norm(x[:3, 3])) + "\n"
    out+=  f"    R GT," + str(Rotation.as_euler(Rotation.from_matrix(x[:3, :3]), 'xyz')) + "\n"
    return out

def append_sample_motions(motions_list, X):
    """
    X is a 4x4 transformation matrix.
    Appends x,y,z, rotvec (r0, r1, r2) to motions_dict.
    Note that this does not change the transform. It does not normalize the motions to be of magnitude 1.
    """
    X = np.array(X)
    x,y,z = X[0:3, 3:4]
    x,y,z = x[0], y[0], z[0]
    r0, r1, r2 = Rotation.as_rotvec(Rotation.from_matrix(X[:3, :3]))
    motions_list.append([x,y,z, r0, r1, r2])


def visualize_batch(batch):
    print(batch['fish0'].shape)
    print(batch['fish1'].shape)

    # Tile all images and show.
    fish1 = np.concatenate([batch['fish0'][i, :, :, :].numpy() for i in range (batch['fish0'].shape[0])], axis = 1)
    fish2 = np.concatenate([batch['fish1'][i, :, :, :].numpy() for i in range (batch['fish1'].shape[0])], axis = 1)
    img = np.concatenate([fish1, fish2], axis = 0)

    # Resize to make smaller.
    img = cv2.resize(img, (0, 0), fx = 0.25, fy = 0.25)

    # Add text to images.
    cv2.putText(img, 'fish1', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.putText(img, 'fish2', (0, 20 + fish2.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Batch Visualization.', img)
    cv2.waitKey(0)

    # Destroy windows.
    cv2.destroyAllWindows()


def sample_to_torch(sample, device = 'cpu'):
    '''
    Convert the sample to torch tensors. Both in type and in shape.
    Input samples are of the form:

    sample = { 'rgb_lcam_front': arraylike(B, S, H, W, 3) uint8,
                ...

               'depth_lcam_front': arraylike(B, S, H, W, 1) float32,
               ...

               'pose_lcam_front': arraylike(B, S, 7) xyz, quat,
               ...

               
               'new_buffer': bool])

    Outputs are images where the types are float32 and the shape is (B, S, C, H, W). The images are also normalized to [0, 1] if their original type was uint8. Similar to what torchvision.transforms.ToTensor() does.
    '''
    for k, v in sample.items():
        if isinstance(v, bool):
            continue

        if isinstance(v, torch.Tensor):
            sample[k] = v.to(device)

            if v.dtype == torch.uint8 and len(v.shape) == 5:
                sample[k] = v.float() / 255.0
                sample[k] = sample[k].permute(0, 1, 4, 2, 3).to(device)

            elif v.dtype == np.uint8 and len(v.shape) == 5:
                sample[k] = torch.from_numpy(v).float() / 255.0
                sample[k] = sample[k].permute(0, 1, 4, 2, 3).to(device)

            elif v.dtype == torch.float32 and len(v.shape) == 5:
                sample[k] = v.permute(0, 1, 4, 2, 3).to(device)

            elif v.dtype == np.float32 and len(v.shape) == 5:
                sample[k] = torch.from_numpy(v).permute(0, 1, 4, 2, 3).to(device)

                
            elif isinstance(v, np.ndarray):
                sample[k] = torch.from_numpy(v).to(device)

    return sample