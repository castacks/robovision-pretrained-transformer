import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import copy
import torchvision.transforms as transforms
import math
import re
from scipy.spatial.transform import Rotation

try:
    torch.meshgrid( torch.rand((3,)), torch.rand((3,)), indexing='ij' )
    
    # No error.
    def torch_meshgrid(*args, indexing='xy'):
        res = torch.meshgrid(*args, indexing=indexing)
        return [ r.contiguous() for r in res ]
    
except TypeError as exc:
    print('meshgrid() compatibility issue detected. The exception is: ')
    print(exc)
    
    # exc must mention the word `indexing`.
    _m = re.search(r'indexing', str(exc))
    assert _m is not None, \
        f'The exception is not the expected one, which should contain the key word "indexing". '
    
    print('Use a customized version of torch.meshgrid(). ')

    def meshgrid_ij(*args):
        res = torch.meshgrid(*args)
        return [ r.contiguous() for r in res ]

    def meshgrid_xy(*args):
        res = torch.meshgrid(*args[::-1])
        return [ r.contiguous() for r in res[::-1] ]

    def torch_meshgrid(*args, indexing='xy'):
        if indexing == 'xy':
            return meshgrid_xy(*args)
        elif indexing == 'ij':
            return meshgrid_ij(*args)
        else:
            raise Exception(f'Expect indexing to be either "xy" or "ij". Got {indexing}. ')

LOCAL_PI = math.pi

def deg2rad(deg):
    global LOCAL_PI
    return deg / 180.0 * LOCAL_PI

class ShapeStruct(object):
    def __init__(self, H, W, C=-1, **kwargs):
        super().__init__()
        
        self.H = round(H)
        self.W = round(W)
        self._C = C
        
    @property
    def shape(self):
        '''
        This funtion is meant to be used with NumPy, PyTorch, etc.
        '''
        return (self.H, self.W)
    
    @property
    def size(self):
        '''
        This function is meant to be used with OpenCV APIs.
        '''
        return (self.W, self.H)
    
    @property
    def shape_numpy(self):
        return np.array( [ self.H, self.W ], dtype=np.int32 )
    
    @staticmethod
    def read_shape_struct(dict_like):
        '''
        Read shape information from a dict-like object.
        '''
        return ShapeStruct( **dict_like ) \
            if not isinstance(dict_like, ShapeStruct) \
            else dict_like

    @property
    def C(self):
        return self._C

    def __str__(self) -> str:
        return f'{{ "H": {self.H}, "W": {self.W}, "C": {self.C} }}'

    def __repr__(self) -> str:
        return f'ShapeStruct(H={self.H}, W={self.W})'
    
    def __eq__(self, other):
        return self.H == other.H and self.W == other.W and self.C == other.C

class SensorModel(object):
    def __init__(self, name, shape_struct, in_to_tensor=False, out_to_numpy=False):
        super().__init__()
        
        self.name = name
        
        self._ss = None # Initial value.
        self.ss = shape_struct # Update self._ss.
        
        self._device = None
        self.in_to_tensor = in_to_tensor
        self.out_to_numpy = out_to_numpy
    
    @staticmethod
    def make_shape_struct_from_repr(shape_struct):
        if isinstance( shape_struct, dict ):
            return ShapeStruct( **shape_struct )
        elif isinstance( shape_struct, ShapeStruct ):
            return shape_struct
        else:
            raise Exception(f'shape_struct must be a dict or ShapeStruct object. Get {type(shape_struct)}')
    
    @property
    def ss(self):
        return self._ss

    @ss.setter
    def ss(self, shape_struct):
        self._ss = SensorModel.make_shape_struct_from_repr(shape_struct)
    
    @property
    def shape(self):
        return self.ss.shape
        
    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, d):
        self._device = d

    def in_wrap(self, x):
        if self.in_to_tensor and not isinstance(x, torch.Tensor):
            return torch.as_tensor(x).to(device=self._device)
        else:
            return x

    def out_wrap(self, x):
        if self.out_to_numpy and isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        else:
            return x

    def __deepcopy__(self, memo):
        '''
        https://stackoverflow.com/questions/57181829/deepcopy-override-clarification#:~:text=In%20%22How%20to%20override%20the%20copy%2Fdeepcopy%20operations%20for,setattr%20%28result%2C%20k%2C%20deepcopy%20%28v%2C%20memo%29%29%20return%20result
        '''
        cls = self.__class__ # Extract the class of the object
        result = cls.__new__(cls) # Create a new instance of the object based on extracted class
        memo[ id(self) ] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo)) # Copy over attributes by copying directly or in case of complex objects like lists for exaample calling the `__deepcopy()__` method defined by them. Thus recursively copying the whole tree of objects.
        return result
    
    def get_rays_wrt_sensor_frame(self, shift=0.5):
        '''
        This function returns the rays shoting from the sensor and a valid mask.
        '''
        raise NotImplementedError()
        
class CameraModel(SensorModel):
    def __init__(self, name, fx, fy, cx, cy, fov_degree, shape_struct, in_to_tensor=False, out_to_numpy=False):
        super(CameraModel, self).__init__(
            name=name, shape_struct=shape_struct, in_to_tensor=in_to_tensor, out_to_numpy=out_to_numpy)

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.fov_degree = fov_degree 
        self.fov_rad = deg2rad( self.fov_degree )
        
        self.padding_mode_if_being_sampled = 'zeros'
        
        # Will be populated once get_valid_mask() is called for the first time.
        self.valid_mask = None

    def _f(self):
        assert self.fx == self.fy
        return self.fx

    @property
    def f(self):
        # _f() is here to be called by child classes.
        return self._f()

    @SensorModel.device.setter
    def device(self, d):
        SensorModel.device.fset(self, d)
        
        if self.valid_mask is not None:
            if isinstance(self.valid_mask, torch.Tensor):
                self.valid_mask = self.valid_mask.to(device=d)

    def pixel_meshgrid(self, shift=0.5, normalized=False, skip_out_wrap=False, flatten=False):
        '''
        Get the meshgrid of the pixel centers.
        shift is applied along the x and y directions.
        If normalized is True, then the pixel coordinates are normalized to [-1, 1].
        '''
        
        H, W = self.shape
        
        x = torch.arange(W, dtype=torch.float32, device=self._device) + shift
        y = torch.arange(H, dtype=torch.float32, device=self._device) + shift
        
        # Compatibility issue with Jetpack 4.6 where
        # Python is 3.6, PyTorch is 1.8.
        xx, yy = torch_meshgrid(x, y, indexing='xy')
        
        xx, yy = xx.contiguous(), yy.contiguous()
        
        if normalized:
            xx = xx / W * 2 - 1
            yy = yy / H * 2 - 1
        
        if flatten:
            xx = xx.view((-1))
            yy = yy.view((-1))
        
        if skip_out_wrap:
            return xx, yy
        else:
            return self.out_wrap(xx), self.out_wrap(yy)
    
    def pixel_coordinates(self, shift=0.5, normalized=False, flatten=False):
        '''
        Get the pixel coordinates.
        shift is appllied along the x and y directions.
        If normalized is True, then the pixel coordinates are normalized to [-1, 1].
        '''
        xx, yy = self.pixel_meshgrid(shift=shift, normalized=normalized, skip_out_wrap=True, flatten=flatten)
        return self.out_wrap( torch.stack( (xx, yy), dim=0 ).contiguous() )

    def pixel_2_ray(self, pixel_coor):
        '''
        Arguments:
        pixel_coor (Tensor): A 2xN Tensor contains the pixel coordinates. 
        
        NOTE: pixel_coor can also have a dimension of Bx2xN, where B is the 
        batch number.
        
        Returns:
        A 3xN Tensor representing the 3D rays. Bx3xN if batched.
        A (N,) Tensor representing the valid mask. BxN if batched.
        '''
        raise NotImplementedError()

    def point_3d_2_pixel(self, point_3d, normalized=False):
        '''
        Arguments:
        point_3d (Tensor): A 3xN Tensor contains 3D point coordinates. 
        normalized (bool): If True, then the returned coordinates are normalized to [-1, 1]
        
        NOTE: point_3d can also have a dimension of Bx3xN, where B is the 
        batch number.
        
        Returns: 
        A 2xN Tensor representing the 2D pixels. Bx2xN if batched.
        A (N,) Tensor representing the valid mask. BxN if batched.
        '''
        raise NotImplementedError()
    
    def get_rays_wrt_sensor_frame(self, shift=0.5):
        '''
        This function returns the rays shooting from the sensor and a valid mask.
        '''
        pixel_coor = self.pixel_coordinates(shift=shift, flatten=True)
        return self.pixel_2_ray( pixel_coor )
    
    def _resize(self, new_shape_struct):
        '''
        In place operation. Child class should overload this method to perform appropriate operations.
        '''
        
        factor_x = new_shape_struct.W / self.ss.W
        factor_y = new_shape_struct.H / self.ss.H
        
        self.fx = self.fx * factor_x
        self.fy = self.fy * factor_y
        
        self.cx = self.cx * factor_x
        self.cy = self.cy * factor_y
        
        self.ss = new_shape_struct
    
    def get_resized(self, new_shape_struct):
        resized = copy.deepcopy(self)
        
        if self.ss == new_shape_struct:
            return resized
        
        # Child class may overload the _resize() method.
        device = self.device
        resized._resize(new_shape_struct)
        resized.device = device
        
        return resized
    
    def get_valid_bounary(self, n_points=100):
        '''
        Get an array of pixel coordinates that represent the boundary of the valid region.
        The result array is ordered.
        '''
        raise NotImplementedError()
    
    def get_valid_mask(self, flatten=False, force_update=False):
        # NOTE: Potential bug if force_update is False and flatten althers between two calls.
        if self.valid_mask is not None and not force_update:
            return self.valid_mask
        
        # Get the pixel coordinates of the pixel centers.
        pixel_coordinates = self.pixel_coordinates( shift=0.5, normalized=False, flatten=True )
        
        # Get the valid mask.
        _, valid_mask = self.pixel_2_ray( pixel_coordinates )
        
        if not flatten:
            valid_mask = valid_mask.view( self.shape )
            
        return valid_mask

class Pinhole(CameraModel):
    def __init__(self, fx, fy, cx, cy, shape_struct, in_to_tensor=False, out_to_numpy=False):
        
        # Compute the FoV from the specified parameters.
        shape_struct = SensorModel.make_shape_struct_from_repr(shape_struct)
        fov_degree = 2 * math.atan2(shape_struct.W, 2 * fx) * 180.0 / LOCAL_PI
        
        super().__init__('Pinhole', fx, fy, cx, cy, fov_degree, shape_struct, in_to_tensor=in_to_tensor, out_to_numpy=out_to_numpy)

        self.set_members_by_shape_struct(shape_struct)
        self.init_depth_to_points(shift=0.5)

    def set_members_by_shape_struct(self, shape_struct):
        # FoV for both longitude (x and width) and latitude (y and height).
        self.fov_degree_longitude = 2 * math.atan2(shape_struct.W, 2 * self.fx) * 180.0 / LOCAL_PI
        self.fov_degree_latitude  = 2 * math.atan2(shape_struct.H, 2 * self.fy) * 180.0 / LOCAL_PI
        # print(f"Created a new Pinhole camera model with lon/lat FoV of {self.fov_degree_longitude, self.fov_degree_latitude} degrees.")
        
        # The (inverse) intrinsics matrix is fixed throughout, keep a copy here.
        self.intrinsics = torch.tensor(
            [ [self.fx, 0      , self.cx ],
              [ 0,      self.fy, self.cy ],
              [ 0,      0      ,      1.0] ] ).to(dtype=torch.float32, device=self._device)

        self.inv_intrinsics = torch.tensor(
            [ [1.0/self.fx, 0      , -self.cx / self.fx],
              [ 0,      1.0/self.fy, -self.cy / self.fy],
              [ 0,      0      ,                    1.0] ] ).to(dtype=torch.float32, device=self._device)

    def _resize(self, new_shape_struct):
        super()._resize(new_shape_struct) # self.ss is updated.
        self.set_members_by_shape_struct(new_shape_struct)

    @CameraModel.device.setter
    def device(self, d):
        CameraModel.device.fset(self, d)
        self.inv_intrinsics = self.inv_intrinsics.to(device=d)
        self.intrinsics = self.intrinsics.to(device=d)

    def pixel_2_ray(self, uv):
        '''
        Arguments:
        uv (Tensor): A 2xN Tensor contains the pixel coordinates. 
        
        NOTE: pixel_coor can also have a dimension of Bx2xN, where B is the 
        batch size.
        
        Returns:
        A 3xN Tensor representing the 3D rays. Bx3xN if batched.
        A (N,) Tensor representing the valid mask. BxN if batched.
        '''

        # Warp to desired datatype.
        uv = self.in_wrap(uv).to(dtype=torch.float32)
        
        # Convert to honmogeneous coordinates.
        uv1 = F.pad(uv, (0, 0, 0, 1), value=1)

        # Convert to camera-frame (metric).
        xyz = self.inv_intrinsics @ uv1

        # Normalize rays to be unit length.
        xyz = xyz / torch.linalg.norm(xyz, dim = -2, keepdims= True)

        # Mask points that are out of field of view.
        # Currently returning a mask of only ones as all pixels are assumed to be with valid values, and the projection out of the image frame of all pixels is valid.
        # The following if-statements match the dimensionality of batched inputs.
        # NOTE(yoraish): why should there be any??
        if len(uv.shape) == 2:
            mask = torch.ones(uv.shape[1], device = self.device)
        if len(uv.shape) == 3:
            mask = torch.ones((uv.shape[0], uv.shape[2]), device = self.device)
        
        return self.out_wrap(xyz), \
               self.out_wrap(mask)
        
    def point_3d_2_pixel(self, point_3d, normalized=False):
        '''
        Arguments:
        point_3d (Tensor): A 3xN Tensor contains 3D point coordinates. 
        normalized (bool): If True, then the returned coordinates are normalized to [-1, 1]
        
        NOTE: point_3d can also have a dimension of Bx3xN, where B is the 
        batch number.
        
        Returns: 
        A 2xN Tensor representing the 2D pixels. Bx2xN if batched.
        A (N,) Tensor representing the valid mask. BxN if batched.
        '''
        point_3d = self.in_wrap(point_3d)

        # Pixel coordinates projected from the world points. 
        uv_unnormalized = self.intrinsics @ point_3d

        # Normalize the homogenous coordinate-points such that their z-value is 1. The expression uv_unnormalized[..., -1:, :] keeps the dimension of the tensor, which is required by the division operation since PyTorch has trouble to broadcast the operation. 
        uv = torch.div(uv_unnormalized, uv_unnormalized[..., -1:, :])

        # Do torch.split results in Bx1XN.
        px, py, _ = torch.split( uv, 1, dim=-2 )

        if normalized:
            # Using shape - 1 is the way for cv2.remap() and align_corners=True of torch.nn.functional.grid_sample().
            # px = px / ( self.ss.W - 1 ) * 2 - 1
            # py = py / ( self.ss.H - 1 ) * 2 - 1
            # Using shape is the way for torch.nn.functional.grid_sample() with align_corners=False.
            px = px / self.ss.W * 2 - 1
            py = py / self.ss.H * 2 - 1

        pixel_coor = torch.cat( (px, py), dim=-2 )

        # Filter the invalid pixels by the image size. Valid mask takes on shape [B] x N
        # If normalized, require the coordinates to be in the range [-1, 1].
        if normalized:
            valid_mask_px = torch.logical_and(px < 1, px > -1)
            valid_mask_py = torch.logical_and(py < 1, py > -1)
        
        # If not normalized, require the coordinates to be in the range [0, W] and [0, H].
        else:
            valid_mask_px = torch.logical_and(px < self.ss.W, px > 0)
            valid_mask_py = torch.logical_and(py < self.ss.H, py > 0)

        valid_mask = torch.logical_and(valid_mask_py, valid_mask_px)

        # This is for the batched dimension.
        valid_mask = valid_mask.squeeze(-2)

        return self.out_wrap(pixel_coor), self.out_wrap(valid_mask)

    def init_depth_to_points(self, shift = 0.5, device = 'cuda'):
        self.G = self.pixel_coordinates(shift = shift, flatten = False).to(device)
        self.x_coef = ( self.G[0] - self.cx ) / self.fx
        self.y_coef = ( self.G[1] - self.cy) / self.fy
        
    def depth_to_points(self, depth):
        x = self.x_coef * depth
        y = self.y_coef * depth
        points = self.out_wrap(torch.stack((x, y, depth), axis=-3))
        return points, self.G

    def __repr__(self) -> str:
        return f'''An instance of Pinhole CameraModel
        Height : {self.ss.shape[0]}
        Width : {self.ss.shape[1]}
        fx : {self.fx}
        fy : {self.fy}
        cx : {self.cx}
        cy : {self.cy}
        FoV degrees (lon/lat, y/x, h/w): {self.fov_degree_longitude}, {self.fov_degree_latitude}
        device: {self._device}'''

    def __str__(self) -> str:
        return f'''Pinhole
        Shape : {self.ss.shape}
        FoV degrees (lon/lat, y/x, h/w): {self.fov_degree_longitude}, {self.fov_degree_latitude}'''


def flow_from_depth_motion(depth0, motion01, camera_model, device = 'cuda'): #, clipping = 1000.
    '''    
    dist0: the depth image. A numpy array or torch tensor of shape (H, W). The depth is in meters.
    motion: The motion vector of the fisheye depth image. xyz, rotvec. Note that the motion is of the NED base frame. This is x forward, y right, z down.
    camera_model: The camera model of the fisheye distance image.
    flow_image_gfp: The path to the flow image to be saved.
    clipping: clip the min/max values of the flow within this range
    '''
    batchsize = 1
    if len(depth0.shape)>2:
        batchsize = depth0.shape[0]

    # The motion should be np.array.
    if type(motion01) == torch.Tensor:
        motion01 = motion01.cpu().numpy()
    motion01 = motion01.reshape((batchsize, 6))

    if type(depth0) == np.ndarray:
        depth0 = torch.from_numpy(depth0).float().to(device)
    else:
        depth0 = depth0.float().to(device)

    points0, G = camera_model.depth_to_points(depth0) # points can be n x 3 x h x w or 3 x h x w

    # # Get the points in the new image frame.
    R1Inv = torch.tensor(Rotation.from_rotvec(motion01[:, 3:]).as_matrix()).float()
    R1 = R1Inv.mT.float().to(device)
    t1 = -R1 @ (torch.from_numpy(motion01[:,:3]).unsqueeze(-1).float().to(device))

    # The coordinates in the world frame.
    XWorld_0 = points0.view(batchsize, 3,-1).roll(1, dims=1) # Coordinates in the NED frame. z-axis pointing downwards.
    # The coordinate of the pixels of the first camera projected in the second camera's frame (NED).
    X1 = R1 @ (XWorld_0) + t1

    # Back to xyz from zxy.
    points1 = X1.roll(2, dims=1) 

    # Get the pixel locations that the new points fall onto.
    G1, valid_mask1 = camera_model.point_3d_2_pixel(points1) # G1 is a 2xH*W tensor.

    # Get the du and dv.
    du = G1[:, 0, :] - G[0, :].view(-1)
    dv = G1[:, 1,:] - G[1,:].view(-1)

    H, W = int(camera_model.ss.H), int(camera_model.ss.W)

    # Reshape.
    if batchsize==1:
        du = du.view((H, W))
        dv = dv.view((H, W))
    else:
        du = du.view((batchsize, H, W))
        dv = dv.view((batchsize, H, W))

    # The flow image.
    flow = torch.stack((du, dv), dim = -3)
    # flow = torch.clamp(flow, -clipping, clipping)
    return flow

def flow16to32(flow16):
    '''
    flow_32b (float32) [-512.0, 511.984375]
    flow_16b (uint16) [0 - 65535]
    flow_32b = (flow16 -32768) / 64
    '''
    flow32 = flow16[:,:,:2].astype(np.float32)
    flow32 = (flow32 - 32768) / 64.0

    mask8 = flow16[:,:,2].astype(np.uint8)
    return flow32, mask8

################################################################################


sys.path.append('..')
from data_management.tartanair_image_reader import TartanAirImageReader
from data_management.transformation import pos_quats2ses
from visualization.visualization_utils import DataVisualizer
# from pinhole_flow import process_single_process, CameraBase
import time

# The directory of the pinhole rgb and depth images.
pin_depth_images_dir = '/home/mihirsharma/depth_lcam_front'
pin_depth0_gfps = [os.path.join(pin_depth_images_dir, f) for f in os.listdir(pin_depth_images_dir) if f.endswith('.png')]
pin_depth0_gfps.sort()

pin_bgr_images_dir = '/home/mihirsharma/image_lcam_front'
pin_bgr0_gfps = [os.path.join(pin_bgr_images_dir, f) for f in os.listdir(pin_bgr_images_dir) if f.endswith('.png')]
pin_bgr0_gfps.sort()

pin_flow_images_dir = '/home/mihirsharma/flow_lcam_front'
pin_flow_gfps = [os.path.join(pin_flow_images_dir, f) for f in os.listdir(pin_flow_images_dir) if f.endswith('.png')]
pin_flow_gfps.sort()

# The directory of the motion vectors.
pose_gfp = '/home/mihirsharma/pose_lcam_front.txt'


pin_camera_model = Pinhole(fx = 320, fy = 320, cx = 320, cy = 320, shape_struct=ShapeStruct(640, 640), out_to_numpy=False)
pin_camera_model.device = 'cuda'
# Get the depth images.
# depth_images = sorted(os.listdir(depth_images

# pin_cam_model_np = CameraBase(focal=320, imageSize=(640,640))

visualizer = DataVisualizer()
gt_poses = np.loadtxt(pose_gfp)
starttime = time.time()
image_list = []
flow_list = []

def build_train_dataset(batch_size = 12, dataset_size = 0.5, crop_size = [1/4, 1/4], downsample_size = 8, standard_deviation = 2):

    for ix in range(round(len(pin_depth0_gfps)*dataset_size-2)):

        #Converting raw png image to tensor
        image = TartanAirImageReader().read_bgr(pin_bgr0_gfps[ix])
        image = torch.tensor(image, device = 'cuda').float().permute(2, 0, 1)

        c, h, w = image.shape

        #Random cropping and token selection
        random_crop_query_location=torch.rand(2).to('cuda') #random crop location
        random_samples_reference_matrix=torch.rand(round(h * crop_size[0] / downsample_size) * round(w * crop_size[1] / downsample_size), 2).to('cuda') # [Crop Area, 2]
        
        
        #Decompressing flow to both flow and mask
        flow16 = cv2.imread(pin_flow_gfps[ix], cv2.IMREAD_UNCHANGED)
        flow16, mask = flow16to32(flow16)
        flow = torch.tensor(flow16, device = 'cuda').float()
        mask = torch.tensor(mask, device = 'cuda').float().unsqueeze(dim=-1) / 100

        #Masked pixels become 0
        masked_flow = torch.mul(flow, mask).permute(2, 0, 1)

        feature_flow = torch.nn.functional.avg_pool2d(masked_flow, kernel_size = downsample_size, stride = downsample_size)

        selective_feature_flow = torch.index_select(feature_flow, )

        feature_correlation = 0 #FIXME

        image_list.append(image.to(torch.float16))
        # flow_list.append(flow0.to(torch.float16))

        torch.cuda.empty_cache()

    images = torch.stack(image_list)

    print(images.shape)

    # return images, feature_correlation

# build_train_dataset()


image = TartanAirImageReader().read_bgr(pin_bgr0_gfps[0])
image = torch.tensor(image, device = 'cuda').float().permute(2, 0, 1)
crop_size = [1/4, 1/4] 
downsample_size = 8
c, h, w = image.shape

feature_map_crop_height = round(h * crop_size[0] / downsample_size)
feature_map_crop_width = round(w * crop_size[1] / downsample_size)


#Random cropping and token selection
random_crop_query_location = torch.round(torch.mul(torch.rand(2), torch.tensor([feature_map_crop_width, feature_map_crop_height]))).to('cuda').to(torch.int32) #random crop location (x, y)
random_samples_reference_matrix=torch.rand(feature_map_crop_height * feature_map_crop_width, 2).to('cuda') # [Crop Area, 2]

random_samples_reference = torch.rand(feature_map_crop_height * feature_map_crop_width).to('cuda') # [Crop Area]


#Decompressing flow to both flow and mask
flow16 = cv2.imread(pin_flow_gfps[0], cv2.IMREAD_UNCHANGED)
flow16, mask = flow16to32(flow16)
flow = torch.tensor(flow16, device = 'cuda').float()
mask = torch.tensor(mask, device = 'cuda').float().unsqueeze(dim=-1) / 100


#Masked pixels become 0
masked_flow = torch.mul(flow, mask).permute(2, 0, 1)

feature_flow = torch.nn.functional.avg_pool2d(masked_flow, kernel_size = downsample_size, stride = downsample_size)

print(feature_flow.shape)

c_ff, h_ff, w_ff = feature_flow.shape

flattened_feature_flow = feature_flow.view(c_ff, h_ff * w_ff)

print(flattened_feature_flow.shape)

selective_masked_flow_indices = torch.round(random_samples_reference * h_ff * w_ff).to(torch.int32)
print(selective_masked_flow_indices.shape)

selective_masked_flow = torch.index_select(flattened_feature_flow, 1, selective_masked_flow_indices)

print(selective_masked_flow.shape)
print(selective_masked_flow_indices.shape)

#Recovers x, y coordinates from 1D positions
reference_x_y_coords =  torch.cat([(selective_masked_flow_indices % feature_map_crop_width).unsqueeze(dim=-1).permute(1, 0), (torch.ceil(selective_masked_flow_indices / feature_map_crop_width).unsqueeze(dim=-1)).permute(1, 0)]).to(torch.int32)

print(reference_x_y_coords[1, 100])

correlation_positions = torch.add(selective_masked_flow, selective_masked_flow_indices)

print(random_crop_query_location.unsqueeze(-1).repeat(1, feature_map_crop_height * feature_map_crop_width).shape)

relative_correlation_positions = torch.sub(correlation_positions, random_crop_query_location.unsqueeze(-1).repeat(1, feature_map_crop_height * feature_map_crop_width))

print(correlation_positions.shape)

print(relative_correlation_positions[1, 250])


