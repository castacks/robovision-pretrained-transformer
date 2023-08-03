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

class MatchingDataset(data.DataSet):
    def __init__(self, aug_params=None, sparse=False,
                 load_occlusion=False,
                 vkitti2=False,
                 ):
        self.augmentor = None
        self.sparse = sparse

        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.vkitti2 = vkitti2

        self.load_occlusion = load_occlusion
        self.occ_list = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])

            if len(np.array(img1).shape) == 2:  # gray image
                img1 = img1.convert('RGB')
                img2 = img2.convert('RGB')

            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None

        if self.sparse:
            if self.vkitti2:
                flow, valid = frame_utils.read_vkitti2_flow(self.flow_list[index])
            else:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])  # [H, W, 2], [H, W]
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        if self.load_occlusion:
            occlusion = frame_utils.read_gen(self.occ_list[index])  # [H, W], 0 or 255 (occluded)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if self.load_occlusion:
            occlusion = np.array(occlusion).astype(np.float32)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                if self.load_occlusion:
                    img1, img2, flow, occlusion = self.augmentor(img1, img2, flow, occlusion=occlusion)
                else:
                    img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.load_occlusion:
            occlusion = torch.from_numpy(occlusion)  # [H, W]

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        # mask out occluded pixels
        if self.load_occlusion:
            # non-occlusion: 0, occlusion: 255
            noc_valid = 1 - occlusion / 255.  # 0 or 1

            return img1, img2, flow, valid.float(), noc_valid.float()

        return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list

        return self

    def __len__(self):
        return len(self.image_list)
    
class TartanAirV2(MatchingDataset):
    def __init__(self,
                 data_dir='datasets/Tartanair',
                 transform=None,
                 ):
        super(TartanAirV2, self).__init__(transform=transform, is_tartanair=True)

        left_files = sorted(glob(data_dir + '/*/*/*/*/image_left/*.png'))
        right_files = sorted(glob(data_dir + '/*/*/*/*/image_right/*.png'))
        disp_files = sorted(glob(data_dir + '/*/*/*/*/depth_left/*.npy'))

        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]
            sample['disp'] = disp_files[i]

            self.samples.append(sample)