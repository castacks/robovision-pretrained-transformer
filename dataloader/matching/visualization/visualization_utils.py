# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from colorama import Fore, Style

import numpy as np
import torch
import imageio


def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 1
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    '''

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)


UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


# from https://github.com/gengshan-y/VCN
def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def save_vis_flow_tofile(flow, output_path):
    vis_flow = flow_to_image(flow)
    from PIL import Image
    img = Image.fromarray(vis_flow)
    img.save(output_path)


def flow_tensor_to_image(flow):
    """Used for tensorboard visualization"""
    flow = flow.permute(1, 2, 0)  # [H, W, 2]
    flow = flow.detach().cpu().numpy()
    flow = flow_to_image(flow)  # [H, W, 3]
    flow = np.transpose(flow, (2, 0, 1))  # [3, H, W]

    return flow

def cat_flow_and_imgs(flows, imgs):

    # Show the optical flow.
    # Stack a channel of ones to the flow.
    h,w,p = 8, 8, 9
    flow = torch.cat((flow.cpu(), torch.zeros( (p, 1,h, w) ) ) , dim = 1)
    flow = flow.permute(0, 2, 3, 1).numpy()

    # Concatenate all to one cv mat.
    show_flow = np.zeros((h * 3, w * 3 * 2, 3))
    show_flow[0:h, 0:w, :] =  flow[0]
    show_flow[0 : h , w : w * 2, :] =  flow[1]
    show_flow[0 : h , w * 2 : w * 3, :] =  flow[2]
    show_flow[h : h * 2, 0 : w, :] =  flow[3]
    show_flow[h : h * 2, w : w * 2, :] =  flow[4]
    show_flow[h : h * 2, w * 2 : w * 3, :] =  flow[5]
    show_flow[h * 2 : h * 3, 0 : w, :] =  flow[6]
    show_flow[h * 2 : h * 3, w : w * 2, :] =  flow[7]
    show_flow[h * 2 : h * 3, w * 2 : w * 3, :] =  flow[8]
    
    # Also the images.
    show_flow[0: h, w * 3 : w * 4, :] = cv2.resize( img0[0].permute(1, 2, 0).cpu().numpy(), (w, h) )
    show_flow[0 : h , w * 4 : w * 5, :] = cv2.resize( img0[1].permute(1, 2, 0).cpu().numpy(), (w, h) )
    show_flow[0 : h, w * 5 : w * 6, :] = cv2.resize( img0[2].permute(1, 2, 0).cpu().numpy(), (w, h) )
    show_flow[h : h * 2, w * 3 : w * 4, :] = cv2.resize( img0[3].permute(1, 2, 0).cpu().numpy(), (w, h) )
    show_flow[h : h * 2, w * 4 : w * 5, :] = cv2.resize( img0[4].permute(1, 2, 0).cpu().numpy(), (w, h) )
    show_flow[h : h * 2, w * 5 : w * 6, :] = cv2.resize( img0[5].permute(1, 2, 0).cpu().numpy(), (w, h) )
    show_flow[h * 2 : h * 3, w * 3 : w * 4, :] = cv2.resize( img0[6].permute(1, 2, 0).cpu().numpy(), (w, h) )
    show_flow[h * 2 : h * 3, w * 4 : w * 5, :] = cv2.resize( img0[7].permute(1, 2, 0).cpu().numpy(), (w, h) )
    show_flow[h * 2 : h * 3, w * 5 : w * 6, :] = cv2.resize( img0[8].permute(1, 2, 0).cpu().numpy(), (w, h) )
    show_flow = show_flow.astype(np.uint8)
    # show_flow = cv2.cvtColor(show_flow, cv2.COLOR_RGB2BGR)
    cv2.imshow('flow', show_flow)
    cv2.waitKey(1)
    return


# Below from TartanAir.
import cv2

class DataVisualizer(object):
    def __init__(self) -> None:
        self.text_bg_color = (230, 130, 10) # BGR
        self.text_color = (70, 200, 230)

    def calculate_angle_distance_from_du_dv(self, du, dv, flagDegree=False):
        a = np.arctan2( dv, du )

        angleShift = np.pi

        if ( True == flagDegree ):
            a = a / np.pi * 180
            angleShift = 180
            # print("Convert angle from radian to degree as demanded by the input file.")

        d = np.sqrt( du * du + dv * dv )

        return a, d, angleShift

    def visflow(self, flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0, add_arrows = False, new_shape = False): 
        """
        Show a optical flow field as the KITTI dataset does.
        Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
        flownp is of shape (H, W, 2)

        The output is a numpy array of shape (H, W, 3) with values in range [0, 255], type uint8.
        """

        ang, mag, _ = self.calculate_angle_distance_from_du_dv( flownp[:, :, 0], flownp[:, :, 1], flagDegree=False )

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


        # Scale to a larger shape before adding arrows, and then scale back.
        # bgr = cv2.resize(bgr, (bgr.shape[1] * 2, bgr.shape[0] * 2))
        flow_h, flow_w, _ = flownp.shape

        if new_shape:
            new_h = new_shape[0]
            new_w = new_shape[1]
        else:
            new_h = flownp.shape[0]
            new_w = flownp.shape[1]
        
        bgr = cv2.resize(bgr, (new_w, new_h))
        flownp = cv2.resize(flownp, (new_w , new_h))

        # Adjust the flow values.
        flownp[:, :, 0] = flownp[:, :, 0] * 2
        flownp[:, :, 1] = flownp[:, :, 1] * 2

        if add_arrows:
            for i in range(0, flownp.shape[0], new_h // 10):
                for j in range(0, flownp.shape[1], new_w // 10):
                    cv2.arrowedLine(bgr, (j, i), (j + int(flownp[i, j, 0]), i + int(flownp[i, j, 1])), (255, 0, 0), 1)
            
        return bgr


    def visdepth(self, depth):
        depthvis = np.clip(400/depth ,0 ,255)
        depthvis = depthvis.astype(np.uint8)
        depthvis = cv2.applyColorMap(depthvis, cv2.COLORMAP_JET)

        return depthvis

    def visdisparity(self, disp, maxthresh = 50):
        dispvis = np.clip(disp,0,maxthresh)
        dispvis = dispvis/maxthresh*255
        dispvis = dispvis.astype(np.uint8)
        dispvis = cv2.applyColorMap(dispvis, cv2.COLORMAP_JET)

        return dispvis


    def visflow_toggle(self, flow, img0, img1, do_scale = False, num_toggles = 20, num_arrows = 10):
        """ 
        Visualize the flow field in a mini animation.
        Args:
            flow(np.array): Optical flow field. (H, W, 2). Type is float32.
            img0(np.array): First image. (H, W, C). 
            img1(np.array): Second image. (H, W, C). 

        """
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        print(Fore.YELLOW + "visflow Flow shape: ", flow.shape, Style.RESET_ALL)
        print(Fore.YELLOW + "visflow Fish0 shape: ", img0.shape, Style.RESET_ALL)
        print(Fore.YELLOW + "visflow Fish1 shape: ", img1.shape, Style.RESET_ALL)

        du, dv = flow[:, :, 0], flow[:, :, 1]
        
        flow_vis = self.visflow(flow)

        if not do_scale:
            # Add arrows.
            for i in range(0, du.shape[0], du.shape[0]// num_arrows):
                for j in range(0, du.shape[1], du.shape[1]// num_arrows):
                    cv2.arrowedLine(flow_vis, (i, j), (int(i + du[j, i]), int(j + dv[j, i])), (0, 0, 255), 2)
                    cv2.arrowedLine(img0, (i , j ), (int(i  + du[j, i] ), int(j  + dv[j, i] )), (0, 0, 255), 2)
                    cv2.circle(img0, (i , j ), 2, (0, 0, 255), 2)

            # Mark the flow endpoints on the second image.
            for i in range(0, du.shape[0], du.shape[0]// num_arrows):
                for j in range(0, du.shape[1], du.shape[1]// num_arrows):
                    cv2.circle(img1, (int(i  + du[j, i] ), int(j  + dv[j, i] )), 1, (0, 0, 255), 2)

        # Resize the flow image.
        else:

            # Scale factor to make it at least 1000x1000.
            scale_factor = int(max(1000 / flow_vis.shape[0], 1000 / flow_vis.shape[1]))

            # Resize the flow image.
            flow_vis = cv2.resize(flow_vis, (flow_vis.shape[1] * scale_factor, flow_vis.shape[0] * scale_factor), interpolation = cv2.INTER_NEAREST)
            img0 = cv2.resize(img0, (img0.shape[1] * scale_factor, img0.shape[0] * scale_factor))
            img1 = cv2.resize(img1, (img1.shape[1] * scale_factor, img1.shape[0] * scale_factor))
            


            # Add arrows.
            for i in range(0, du.shape[0], du.shape[0]// num_arrows):
                for j in range(0, du.shape[1], du.shape[1]// num_arrows):
                    cv2.arrowedLine(flow_vis, (i * scale_factor, j * scale_factor), (int(i * scale_factor + du[j, i] * scale_factor), int(j * scale_factor + dv[j, i] * scale_factor)), (0, 0, 255), 2)
                    cv2.arrowedLine(img0, (i * scale_factor, j * scale_factor), (int(i * scale_factor + du[j, i] * scale_factor), int(j * scale_factor + dv[j, i] * scale_factor)), (0, 0, 255), 1)
                    cv2.circle(img0, (i * scale_factor, j * scale_factor), 2, (0, 0, 255), 2)

            # Mark the flow endpoints on the second image.
            for i in range(0, du.shape[0], du.shape[0]// num_arrows):
                for j in range(0, du.shape[1], du.shape[1]// num_arrows):
                    cv2.circle(img1, (int(i * scale_factor + du[j, i] * scale_factor), int(j * scale_factor + dv[j, i] * scale_factor)), 1, (0, 0, 255), 2)

        # Show img0 and img1 alternating continuously. Until user asks to stop.
    

        for k in range(num_toggles):
            cv2.imshow('img01', img0)
            cv2.waitKey(800)
            cv2.imshow('img01', img1)
            cv2.waitKey(800)
            print(k)



    def visflow_compare(self, flow0, flow1, valid_mask, cov = None, img = None, num_arrows = 10, do_scale = True):
        """ 
        Comapre two flow fields. 
        Args:
            flow0(np.array): Optical flow field of ground truth. (H, W, 2). Type is float32.
            flow1(np.array): Optical flow field of prediction. (H, W, 2). Type is float32.
            valid_mask(np.array): Mask of valid pixels. (H, W). Type is uint8.
            cov(np.array): Covariance matrix of the flow field. (H, W, 2). Type is float32.
        """
        du0, dv0 = flow0[:, :, 0], flow0[:, :, 1]
        du1, dv1 = flow1[:, :, 0], flow1[:, :, 1]

        flow0_vis = self.visflow(flow0)
        flow1_vis = self.visflow(flow1)
        if type(img) == type(None):
            flow_compare_vis = np.ones_like(flow0_vis) * 255
        else:
            flow_compare_vis = img.copy() * 0.3 + 0.7 * np.ones_like(flow0_vis) * 255

        # Apply the valid mask.
        flow0_vis = flow0_vis * valid_mask[:, :, np.newaxis]
        flow1_vis = flow1_vis * valid_mask[:, :, np.newaxis]
        flow_compare_vis = flow_compare_vis * valid_mask[:, :, np.newaxis]


        # Output three images. The first flow, the second flow, and the overlay of the two flows.

        if not do_scale:
            scale_factor = 1
            

        # Scale factor to make it at least 1000x1000.
        scale_factor = int(max(1000 / flow0_vis.shape[0], 1000 / flow0_vis.shape[1]))

        # Resize the flow image.
        flow_compare_vis = cv2.resize(flow_compare_vis, (flow_compare_vis.shape[1] * scale_factor, flow_compare_vis.shape[0] * scale_factor), interpolation = cv2.INTER_NEAREST)
        # img0 = cv2.resize(img0, (img0.shape[1] * scale_factor, img0.shape[0] * scale_factor))
        # img1 = cv2.resize(img1, (img1.shape[1] * scale_factor, img1.shape[0] * scale_factor))

        # Add arrows.
        for i in range(0, du0.shape[0], du0.shape[0]// num_arrows):
            for j in range(0, du0.shape[1], du0.shape[1]// num_arrows):

                # Check if this pixel is valid.
                if valid_mask[j, i] == 0:
                    continue

                # Arrow for the GT flow.
                cv2.arrowedLine(flow_compare_vis, (i * scale_factor, j * scale_factor), (int(i * scale_factor + du0[j, i] * scale_factor), int(j * scale_factor + dv0[j, i] * scale_factor)), (0, 0, 0), 2)

                # Arrow for the predicted flow.
                cv2.arrowedLine(flow_compare_vis, (i * scale_factor, j * scale_factor), (int(i * scale_factor + du1[j, i] * scale_factor), int(j * scale_factor + dv1[j, i] * scale_factor)), (0, 0, 255), 2)

                cv2.circle(flow_compare_vis, (i * scale_factor, j * scale_factor), 2, (0, 0, 255), 2)

                # The epe of the flow.
                epe = np.sqrt((du0[j, i] - du1[j, i])**2 + (dv0[j, i] - dv1[j, i])**2)

                # If a covariance is not specified, add a circle to indicate the epe.
                if type(cov) == type(None):
                    cv2.circle(flow_compare_vis, (int(i * scale_factor), int(j * scale_factor )), int(epe * scale_factor), (0, 0, 255), 1)

                # If a covariance is specified, add an ellipse to indicate the epe.
                else:
                    # The covariance of the flow.
                    cov_ij = cov[j, i, :] # Only have the diagonal.

                    # The major and minor axis of the ellipse.
                    major_axis = np.sqrt(cov_ij[0]) * scale_factor
                    minor_axis = np.sqrt(cov_ij[1]) * scale_factor

                    # Draw the ellipse.
                    cv2.ellipse(flow_compare_vis, (int(i * scale_factor), int(j * scale_factor)), (int(major_axis), int(minor_axis)), 0, 0, 360, (255, 0, 0), 1)

        # To RGB.
        flow_compare_vis = flow_compare_vis[:, :, ::-1]

        return flow_compare_vis

    def get_images_and_flow_stacked(self, images, flows, image_labels, flow_labels):


        '''
        Get an images showing bgr images and flows in a grid.
        Args:
            images(list): List of images. Each entry is a (P, C, H, W) torch tensor. The format is BGR. Pixels are [0, 1] float32.
            flows(list): List of flows. Each entry is a (P, 2, H, W) torch tensor. Pixels are [du, dv]. Pixels are floats.
            image_labels(list): List of image labels.
            flow_labels(list): List of flow labels.
        '''

        print(Fore.CYAN + "Visualizing the flows and pinholes." + Style.RESET_ALL)

        # Number of rows and columns.
        num_rows = len(images) + len(flows)
        num_cols = images[0].shape[0]

        # Image size.
        # img_h, img_w = images[0].shape[-2:]
        img_h, img_w = 500, 500

        # Create the output image.
        outimg = np.zeros((num_rows * img_h, num_cols * img_w, 3), dtype=np.uint8)

        # Add the images.
        for i in range(len(images)):
            for j in range(num_cols):
                img = images[i][j].permute(1,2,0).detach().cpu().numpy() * 255.0
                img = cv2.resize(img, (img_w, img_h), interpolation = cv2.INTER_NEAREST)
                outimg[i * img_h: (i + 1) * img_h, j * img_w: (j + 1) * img_w, :] = img

        # Add the flows.
        for i in range(len(flows)):
            for j in range(num_cols):
                outimg[(i + len(images)) * img_h: (i + len(images) + 1) * img_h, j * img_w: (j + 1) * img_w, :] = \
                    self.visflow(flows[i][j].permute(1,2,0).detach().cpu().numpy(), add_arrows= True, new_shape=(img_h, img_w))

        # Add the labels.
        for i in range(len(image_labels)):
            cv2.putText(outimg, image_labels[i], (0, (i + 1) * img_h), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        for i in range(len(flow_labels)):
            cv2.putText(outimg, flow_labels[i], (0, (i + len(images) + 1) * img_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return outimg

    def visualize_images_and_flows_stacked(self, images, flows, image_labels, flow_labels):
        '''
        Visualize images and flows in a grid.
        Args:
            images(list): List of images. Each entry is a (P, C, H, W) torch tensor. The format is BGR. Pixels are [0, 1] float32.
            flows(list): List of flows. Each entry is a (P, 2, H, W) torch tensor. Pixels are [du, dv]. Pixels are floats.
            image_labels(list): List of image labels.
            flow_labels(list): List of flow labels.
        '''
        outimg = self.get_images_and_flow_stacked(images, flows, image_labels, flow_labels)
        # Show the image.
        cv2.namedWindow("Flow", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions.
        cv2.imshow('Flow', outimg)
        cv2.waitKey(0)


    def create_gif_from_image_lists(self, image_lists, image_labels, gif_fname, fps=1):
        '''
        Create a gif from a list of images.
        Args:
            image_lists(list): List of images. Each entry is a (H, W, C) np array. The format is BGR. Pixels are [0, 255] uint8.
            image_labels(list): List of image labels.
            gif_name(str): Name of the gif.
            fps(int): Frames per second.
        '''

        # Create the output image. Concatenated horizontally.
        # A black bar.
        barwidth = 10
        black_bar = np.array([0, 0, 0] * image_lists[0][0].shape[0] * barwidth).reshape((-1, barwidth, 3))
        frames = []
        for i in range(len(image_lists[0])):
            frame_images = [l[i] for l in image_lists]

            # Add a bar between images.
            frame_images = [np.concatenate([f, black_bar], axis=1) for f in frame_images[:-1]] + [frame_images[-1]]

            frame = np.concatenate(frame_images, axis=1)

            # Frame to uint8. Range [0, 255].
            frame = (frame).astype(np.uint8)

            # Add the frame.
            frames.append(frame)

        # Create gif.
        imageio.mimsave(gif_fname, frames, fps=fps)

    def visualize_cov(self, img, cov):
        """

        Args:
            img (np.array): Image. (H, W, C)
            cov (np.array): Covariance matrix. (H, W, 2)
        """

        # Convert to RGB.
        img = img[:, :, ::-1]

        # Scale the image.
        scale_factor = 2
        img = cv2.resize(img, (img.shape[1] * scale_factor, img.shape[0] * scale_factor), interpolation = cv2.INTER_NEAREST)

        # Scale the covariance matrix.
        cov = cv2.resize(cov, (cov.shape[1] * scale_factor, cov.shape[0] * scale_factor), interpolation = cv2.INTER_NEAREST)

        # Draw the covariance matrix.
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                cov_ij = cov[i, j, :]
                cov_ij = cov_ij / np.linalg.norm(cov_ij)
                cv2.arrowedLine(img, (j, i), (int(j + cov_ij[0] * 10), int(i + cov_ij[1] * 10)), (0, 0, 255), 1)

        # Show the image.
        cv2.namedWindow("Covariance", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions.
        cv2.imshow('Covariance', img)
        cv2.waitKey(0)