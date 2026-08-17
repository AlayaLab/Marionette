"""
自定义 Dataset：同时支持 Pose 控制视频 + Camera 相机轨迹

使用方式：
    accelerate launch scripts/wan2.2_fun/train_control.py \
        --dataset_module videox_fun.data.dataset_pose_camera \
        --dataset_class ImageVideoPoseCameraDataset \
        --train_mode control_pose_camera_ref \
        ...

JSON 格式支持两种相机数据格式：

1. CameraCtrl .txt 格式（原有）：
   {
     "camera_file_path": "camera/video_001_camera.txt"
   }

2. NPZ 格式（推荐，更灵活）：
   {
     "camera_file_path": "camera/video_001_camera.npz"
   }
   
   NPZ 文件需包含以下 key：
   - c2w: [N, 4, 4] Camera-to-World 矩阵
   - intrinsics: [N, 4] 内参 [fx, fy, cx, cy]，单位为像素
   
   可选 key：
   - w2c: [N, 4, 4] World-to-Camera 矩阵（如果没有 c2w，会从 w2c 反推）
   - K: [N, 3, 3] 内参矩阵（如果没有 intrinsics，会从 K 提取）
   - original_size: [H, W] 原始图像尺寸（用于缩放内参）
"""

import csv
import gc
import json
import os
import random
from random import shuffle

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader
from einops import rearrange
from func_timeout import FunctionTimedOut, func_timeout
from packaging import version as pver
from PIL import Image
from torch.utils.data.dataset import Dataset

from .utils import (VIDEO_READER_TIMEOUT, VideoReader_contextmanager,
                    get_video_reader_batch, padding_image,
                    resize_frame, resize_image_with_target_area,
                    Camera, get_relative_pose, ray_condition)


# ==================== 相机参数处理工具函数 ====================

def custom_meshgrid(*args):
    """Create meshgrid compatible with different PyTorch versions."""
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def get_relative_pose_from_c2w(c2ws: np.ndarray) -> np.ndarray:
    """
    从 c2w 矩阵序列计算相对位姿（相对于第一帧）
    
    Args:
        c2ws: [N, 4, 4] Camera-to-World 矩阵
        
    Returns:
        [N, 4, 4] 相对位姿矩阵
    """
    # 计算第一帧的 w2c
    w2c_first = np.linalg.inv(c2ws[0])
    
    # 目标坐标系（第一帧为单位矩阵）
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    # 转换矩阵
    abs2rel = target_cam_c2w @ w2c_first
    
    # 计算相对位姿
    ret_poses = [target_cam_c2w]
    for c2w in c2ws[1:]:
        ret_poses.append(abs2rel @ c2w)
    
    return np.array(ret_poses, dtype=np.float32)


def ray_condition_from_intrinsics(K: torch.Tensor, c2w: torch.Tensor, 
                                   H: int, W: int, device='cpu') -> torch.Tensor:
    """
    从内参和 c2w 计算 Plücker 坐标
    
    Args:
        K: [B, N, 4] 内参 [fx, fy, cx, cy]，单位为像素
        c2w: [B, N, 4, 4] Camera-to-World 矩阵
        H, W: 目标图像尺寸
        device: 计算设备
        
    Returns:
        [B, N, H, W, 6] Plücker 坐标
    """
    B = K.shape[0]
    
    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5
    
    fx, fy, cx, cy = K.chunk(4, dim=-1)  # [B, N, 1]
    
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)
    
    directions = torch.stack((xs, ys, zs), dim=-1)  # [B, N, HW, 3]
    directions = directions / directions.norm(dim=-1, keepdim=True)
    
    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # [B, N, HW, 3]
    rays_o = c2w[..., :3, 3]  # [B, N, 3]
    rays_o = rays_o[:, :, None].expand_as(rays_d)
    
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)
    
    return plucker


def load_camera_from_npz(npz_path: str, target_width: int, target_height: int,
                         batch_index: np.ndarray = None) -> torch.Tensor:
    """
    从 NPZ 文件加载相机参数并转换为 Plücker 坐标
    
    NPZ 文件格式：
    - c2w: [N, 4, 4] Camera-to-World 矩阵
    - intrinsics: [N, 4] 内参 [fx, fy, cx, cy]，单位为像素
    
    可选：
    - w2c: [N, 4, 4] 如果没有 c2w，从 w2c 反推
    - K: [N, 3, 3] 如果没有 intrinsics，从 K 矩阵提取
    - original_size: [H, W] 原始图像尺寸
    
    Args:
        npz_path: NPZ 文件路径
        target_width, target_height: 目标图像尺寸
        batch_index: 采样帧索引
        
    Returns:
        [N_sampled, H, W, 6] Plücker 坐标
    """
    data = np.load(npz_path)
    
    # 获取 c2w 矩阵
    if 'c2w' in data:
        c2ws = data['c2w'].astype(np.float32)
    elif 'w2c' in data:
        w2cs = data['w2c'].astype(np.float32)
        c2ws = np.linalg.inv(w2cs)
    else:
        raise ValueError(f"NPZ 文件必须包含 'c2w' 或 'w2c' 矩阵: {npz_path}")
    
    # 获取内参
    if 'intrinsics' in data:
        intrinsics = data['intrinsics'].astype(np.float32)  # [N, 4]
    elif 'K' in data:
        K = data['K'].astype(np.float32)  # [N, 3, 3]
        # 从 K 矩阵提取 fx, fy, cx, cy
        intrinsics = np.stack([
            K[:, 0, 0],  # fx
            K[:, 1, 1],  # fy
            K[:, 0, 2],  # cx
            K[:, 1, 2],  # cy
        ], axis=1)
    else:
        raise ValueError(f"NPZ 文件必须包含 'intrinsics' 或 'K' 矩阵: {npz_path}")
    
    # 获取原始图像尺寸（用于缩放内参）
    if 'original_size' in data:
        orig_h, orig_w = data['original_size']
    else:
        # 假设内参已经是针对目标尺寸的
        orig_h, orig_w = target_height, target_width
    
    # 根据 batch_index 采样
    if batch_index is not None:
        c2ws = c2ws[batch_index]
        intrinsics = intrinsics[batch_index]
    
    # 缩放内参到目标尺寸
    scale_x = target_width / orig_w
    scale_y = target_height / orig_h
    intrinsics[:, 0] *= scale_x  # fx
    intrinsics[:, 1] *= scale_y  # fy
    intrinsics[:, 2] *= scale_x  # cx
    intrinsics[:, 3] *= scale_y  # cy
    
    # 计算相对位姿
    rel_c2ws = get_relative_pose_from_c2w(c2ws)
    
    # 转换为 tensor
    K_tensor = torch.from_numpy(intrinsics)[None]  # [1, N, 4]
    c2w_tensor = torch.from_numpy(rel_c2ws)[None]  # [1, N, 4, 4]
    
    # 计算 Plücker 坐标
    plucker = ray_condition_from_intrinsics(
        K_tensor, c2w_tensor, target_height, target_width
    )  # [1, N, H, W, 6]
    
    return plucker[0]  # [N, H, W, 6]


def load_camera_from_txt(txt_path: str, target_width: int, target_height: int,
                         video_length: int, batch_index: np.ndarray = None,
                         original_pose_width: int = 1280, 
                         original_pose_height: int = 720) -> torch.Tensor:
    """
    从 CameraCtrl 格式的 TXT 文件加载相机参数
    
    TXT 格式（每行）：
    frame_idx fx fy cx cy timestep w2c_mat[12个值：3x4矩阵按行展开]
    
    其中 fx, fy, cx, cy 是归一化值（0-1）
    """
    with open(txt_path, 'r') as f:
        poses = f.readlines()
    
    poses = [pose.strip().split(' ') for pose in poses[1:]]
    cam_params = [[float(x) for x in pose] for pose in poses]
    
    # 如果有采样索引，先插值到视频长度再采样
    if batch_index is not None:
        cam_params_array = np.array(cam_params)
        cam_params_tensor = torch.from_numpy(cam_params_array).unsqueeze(0).unsqueeze(0)
        cam_params_interp = F.interpolate(
            cam_params_tensor,
            size=(video_length, cam_params_tensor.size(3)),
            mode='bilinear',
            align_corners=True
        )[0][0]
        cam_params = [cam_params_interp[idx].numpy().tolist() for idx in batch_index]
    
    # 解析相机参数
    cam_objs = [Camera(cam_param) for cam_param in cam_params]
    
    sample_wh_ratio = target_width / target_height
    pose_wh_ratio = original_pose_width / original_pose_height
    
    if pose_wh_ratio > sample_wh_ratio:
        resized_ori_w = target_height * pose_wh_ratio
        for cam_obj in cam_objs:
            cam_obj.fx = resized_ori_w * cam_obj.fx / target_width
    else:
        resized_ori_h = target_width / pose_wh_ratio
        for cam_obj in cam_objs:
            cam_obj.fy = resized_ori_h * cam_obj.fy / target_height
    
    intrinsic = np.asarray([
        [cam_obj.fx * target_width,
         cam_obj.fy * target_height,
         cam_obj.cx * target_width,
         cam_obj.cy * target_height]
        for cam_obj in cam_objs
    ], dtype=np.float32)
    
    K = torch.as_tensor(intrinsic)[None]  # [1, N, 4]
    c2ws = get_relative_pose(cam_objs)
    c2ws = torch.as_tensor(c2ws)[None]  # [1, N, 4, 4]
    
    plucker = ray_condition(K, c2ws, target_height, target_width, device='cpu')
    # [1, N, H, W, 6]
    
    return plucker[0]  # [N, H, W, 6]


def load_camera_data(camera_path: str, target_width: int, target_height: int,
                     video_length: int, batch_index: np.ndarray = None) -> torch.Tensor:
    """
    统一的相机数据加载接口，自动识别文件格式
    
    Args:
        camera_path: 相机文件路径（.txt 或 .npz）
        target_width, target_height: 目标图像尺寸
        video_length: 视频总帧数
        batch_index: 采样帧索引
        
    Returns:
        [N, H, W, 6] Plücker 坐标
    """
    if camera_path.lower().endswith('.npz'):
        return load_camera_from_npz(camera_path, target_width, target_height, batch_index)
    elif camera_path.lower().endswith('.txt'):
        return load_camera_from_txt(camera_path, target_width, target_height, 
                                     video_length, batch_index)
    else:
        raise ValueError(f"不支持的相机文件格式: {camera_path}")


# ==================== Dataset 定义 ====================

class ImageVideoPoseCameraDataset(Dataset):
    """
    同时支持 Pose 控制视频和 Camera 相机轨迹的 Dataset
    
    JSON 格式示例：
    [
      {
        "file_path": "train/00000001.mp4",           # 目标视频
        "control_file_path": "control/pose.mp4",     # Pose 控制视频
        "camera_file_path": "camera/camera.npz",     # Camera 轨迹文件（.npz 或 .txt）
        "text": "A person walking.",                  # 文本描述
        "type": "video"
      }
    ]
    
    返回字段：
    - pixel_values: [T, C, H, W]，目标视频
    - control_pixel_values: [T, C, H, W]，Pose 控制视频
    - control_camera_values: [T, H, W, 6] 或 list，相机 Plücker 坐标
    - subject_image: np.ndarray 或 None，主体参考图像
    - text: str，文本描述
    - data_type: "video" 或 "image"
    """
    
    def __init__(
        self,
        ann_path,
        data_root=None,
        video_sample_size=512,
        video_sample_stride=4,
        video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.1,
        video_length_drop_end=0.9,
        enable_inpaint=False,
        enable_camera_info=True,  # 保持接口兼容，但本 Dataset 始终启用
        return_file_name=False,
        enable_subject_info=False,
        padding_subject_info=True,
    ):
        # 加载标注文件
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
        else:
            raise ValueError(f"Unsupported annotation file format: {ann_path}")
        
        self.data_root = data_root
        
        # 平衡图像和视频数量
        if video_repeat > 0:
            self.dataset = []
            for data in dataset:
                if data.get('type', 'image') != 'video':
                    self.dataset.append(data)
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        else:
            self.dataset = dataset
        del dataset
        
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.enable_camera_info = True  # 本 Dataset 始终启用 camera
        self.enable_subject_info = enable_subject_info
        self.padding_subject_info = padding_subject_info
        self.return_file_name = return_file_name
        
        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end
        
        # 视频参数
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose([
            transforms.Resize(min(self.video_sample_size)),
            transforms.CenterCrop(self.video_sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.video_transforms_camera = transforms.Compose([
            transforms.Resize(min(self.video_sample_size)),
            transforms.CenterCrop(self.video_sample_size)
        ])
        
        # 图像参数
        self.image_sample_size = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))
    
    def _get_full_path(self, relative_path):
        """获取完整路径"""
        if relative_path is None:
            return None
        if self.data_root is None:
            return relative_path
        return os.path.join(self.data_root, relative_path)
    
    def _load_subject_images(self, object_file_paths, visual_height, visual_width):
        """加载主体参考图像"""
        if not object_file_paths:
            return None
        
        shuffle(object_file_paths)
        subject_images = []
        
        for i in range(min(len(object_file_paths), 4)):
            subject_path = self._get_full_path(object_file_paths[i])
            subject_image = Image.open(subject_path).convert('RGB')
            
            if self.padding_subject_info:
                img = padding_image(subject_image, visual_width, visual_height)
            else:
                img = resize_image_with_target_area(subject_image, 1024 * 1024)
            
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            subject_images.append(np.array(img))
        
        if self.padding_subject_info:
            return np.array(subject_images)
        else:
            return subject_images
    
    def get_batch(self, idx):
        """获取单个样本"""
        data_info = self.dataset[idx % len(self.dataset)]
        video_id = data_info['file_path']
        text = data_info['text']
        
        if data_info.get('type', 'image') == 'video':
            video_path = self._get_full_path(video_id)
            
            with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
                video_total_length = len(video_reader)
                
                min_sample_n_frames = min(
                    self.video_sample_n_frames,
                    int(video_total_length * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No frames in video: {video_path}")
                
                video_length = int(self.video_length_drop_end * video_total_length)
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx = random.randint(
                    int(self.video_length_drop_start * video_total_length), 
                    video_length - clip_length
                ) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)
                
                # 1. 加载目标视频
                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error: {e}")
                
                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    pixel_values = self.video_transforms(pixel_values)
                
                # Random text drop
                if random.random() < self.text_drop_ratio:
                    text = ''
            
            # 2. 加载 Pose 控制视频
            control_video_path = self._get_full_path(data_info.get('control_file_path'))
            if control_video_path and os.path.exists(control_video_path):
                with VideoReader_contextmanager(control_video_path, num_threads=2) as control_video_reader:
                    try:
                        sample_args = (control_video_reader, batch_index)
                        control_pixel_values = func_timeout(
                            VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                        )
                        resized_frames = []
                        for i in range(len(control_pixel_values)):
                            frame = control_pixel_values[i]
                            resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                            resized_frames.append(resized_frame)
                        control_pixel_values = np.array(resized_frames)
                    except FunctionTimedOut:
                        raise ValueError(f"Read control video {idx} timeout.")
                    except Exception as e:
                        raise ValueError(f"Failed to extract frames from control video. Error: {e}")
                    
                    if not self.enable_bucket:
                        control_pixel_values = torch.from_numpy(control_pixel_values).permute(0, 3, 1, 2).contiguous()
                        control_pixel_values = control_pixel_values / 255.
                        control_pixel_values = self.video_transforms(control_pixel_values)
            else:
                # 如果没有 pose 控制视频，使用全零
                if not self.enable_bucket:
                    control_pixel_values = torch.zeros_like(pixel_values)
                else:
                    control_pixel_values = np.zeros_like(pixel_values)
            
            # 3. 加载 Camera 轨迹（新增字段 camera_file_path）
            camera_path = self._get_full_path(data_info.get('camera_file_path'))
            if camera_path and os.path.exists(camera_path):
                try:
                    # 使用统一接口加载相机数据
                    control_camera_values = load_camera_data(
                        camera_path,
                        target_width=self.video_sample_size[1],
                        target_height=self.video_sample_size[0],
                        video_length=video_total_length,
                        batch_index=batch_index
                    )
                    # [N, H, W, 6] -> [N, 6, H, W]
                    if not self.enable_bucket:
                        control_camera_values = control_camera_values.permute(0, 3, 1, 2).contiguous()
                        control_camera_values = self.video_transforms_camera(control_camera_values)
                    else:
                        control_camera_values = control_camera_values.numpy()
                except Exception as e:
                    print(f"Warning: Failed to load camera data from {camera_path}: {e}")
                    control_camera_values = None
            else:
                control_camera_values = None
            
            # 4. 加载主体参考图像（可选）
            subject_image = None
            if self.enable_subject_info:
                if not self.enable_bucket:
                    visual_height, visual_width = pixel_values.shape[-2:]
                else:
                    visual_height, visual_width = pixel_values.shape[1:3]
                
                object_file_paths = data_info.get('object_file_path', [])
                subject_image = self._load_subject_images(object_file_paths, visual_height, visual_width)
            
            return pixel_values, control_pixel_values, subject_image, control_camera_values, text, "video"
        
        else:
            # 图像处理逻辑
            image_path = self._get_full_path(data_info['file_path'])
            image = Image.open(image_path).convert('RGB')
            
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)
            
            if random.random() < self.text_drop_ratio:
                text = ''
            
            # 图像的控制信号
            control_image_path = self._get_full_path(data_info.get('control_file_path'))
            if control_image_path and os.path.exists(control_image_path):
                control_image = Image.open(control_image_path).convert('RGB')
                if not self.enable_bucket:
                    control_image = self.image_transforms(control_image).unsqueeze(0)
                else:
                    control_image = np.expand_dims(np.array(control_image), 0)
            else:
                if not self.enable_bucket:
                    control_image = torch.zeros_like(image)
                else:
                    control_image = np.zeros_like(image)
            
            # 图像模式下 camera 为 None
            control_camera_values = None
            
            # 主体参考图像
            subject_image = None
            if self.enable_subject_info:
                if not self.enable_bucket:
                    visual_height, visual_width = image.shape[-2:]
                else:
                    visual_height, visual_width = image.shape[1:3]
                
                object_file_paths = data_info.get('object_file_path', [])
                subject_image = self._load_subject_images(object_file_paths, visual_height, visual_width)
            
            return image, control_image, subject_image, control_camera_values, text, 'image'
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")
                
                pixel_values, control_pixel_values, subject_image, control_camera_values, text, data_type = self.get_batch(idx)
                
                sample["pixel_values"] = pixel_values
                sample["control_pixel_values"] = control_pixel_values
                sample["subject_image"] = subject_image
                sample["text"] = text
                sample["data_type"] = data_type
                sample["idx"] = idx
                
                # 始终返回 camera 信息
                sample["control_camera_values"] = control_camera_values
                
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(data_info.get('file_path', ''))
                
                if len(sample) > 0:
                    break
                    
            except Exception as e:
                print(f"Error loading sample {idx}: {e}, {self.dataset[idx % len(self.dataset)]}")
                idx = random.randint(0, self.length - 1)
        
        return sample
