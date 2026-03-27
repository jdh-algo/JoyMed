import os 
import glob
import numpy as np
from typing import Any, Dict, List, NewType, Optional, Tuple, Union
from PIL import Image
import nibabel as nib
import re
from collections import Counter
import cv2

def load_m3d_image_folder(
    image_path: Union[str, List[str], Image.Image, List[Image.Image]],
) -> np.ndarray:
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]
    if not image_files:
        raise ValueError(f"No image files found in directory: {image_path}")
    
    # sort and reorder
    image_files.sort(key=_natural_key)
    image_files = reorder_interlaced_files(image_files)

    images_3d = load_image_slices_strict(image_path, image_files).astype(np.uint8)

    metadata = {
        'spacing': (1.0, 1.0, 1.0),
        'shape': images_3d.shape,
        'dtype': str(images_3d.dtype),
    }

    return images_3d, metadata


def load_images_folder(image_path: str) -> List[Image.Image]:
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]
    if not image_files:
        raise ValueError(f"No image files found in directory: {image_path}")

    image_files.sort(key=_natural_key)
    image_files = reorder_interlaced_files(image_files)

    images = load_images_PIL(image_path, image_files)

    if len(images) == 1:
        images = [images[0], images[0]]

    return images

def validate_and_filter_slices(images_3d, min_slices=1):
    if len(images_3d) < min_slices:
        return None

    if isinstance(images_3d[0], np.ndarray):
        # 只比较空间维度 (H, W)
        spatial_shapes = [img.shape[:2] for img in images_3d]
        shape_counts = Counter(spatial_shapes)
        most_common_hw, _ = shape_counts.most_common(1)[0]  # (H, W)

        filtered_images = []
        target_h, target_w = most_common_hw

        for img in images_3d:
            current_hw = img.shape[:2]
            if current_hw != most_common_hw:
                # print(f"=== resize image {current_hw} -> {most_common_hw}")
                # 注意：cv2.resize 使用 (width, height)
                resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                filtered_images.append(resized)
            else:
                filtered_images.append(img)
    elif isinstance(images_3d[0], Image.Image):
        spatial_shapes = [(img.height, img.width) for img in images_3d]  # 明确 H, W
        # print("=== spatial_shapes: ", spatial_shapes)

        shape_counts = Counter(spatial_shapes)
        most_common_hw, _ = shape_counts.most_common(1)[0]  # (H, W)
        # print("=== most_common_hw: ", most_common_hw)

        target_h, target_w = most_common_hw

        filtered_images = []
        for img in images_3d:
            h, w = img.height, img.width
            if (h, w) != (target_h, target_w):
                resized = img.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
                filtered_images.append(resized)
            else:
                filtered_images.append(img)
    else:
        raise ValueError(f"images_3d must be List[np.ndarray] or List[Image.Image], but found type {type(images_3d)}")

    print(f"切片验证成功: {len(filtered_images)}张有效切片")
    return filtered_images


def load_image_slices_strict(image_path: str, image_files: List[str]) -> np.ndarray:
    images_3d = []
    for image_file in image_files:
        img = Image.open(os.path.join(image_path, image_file))
        img = img.convert("L")
        img_array = np.array(img, dtype=np.uint8)
        images_3d.append(img_array)

    images_3d = validate_and_filter_slices(images_3d)
    stacked_array = np.stack(images_3d, axis=0)
    images_3d = np.flip(stacked_array, axis=1)
    images_3d = np.transpose(images_3d, (2, 1, 0))
    images_3d = np.flip(images_3d, axis=0)
    return images_3d

def load_images_PIL(image_path: str, image_files: List[str]) -> List[Image.Image]:
    images = []
    for image_file in image_files:
        img = Image.open(os.path.join(image_path, image_file)).convert("RGB")
        images.append(img)
    images = validate_and_filter_slices(images)
    return images

def _natural_key(filename):
    """
    自然排序key函数
    正确处理: 1.png, 2.png, ..., 10.png, 11.png
    """
    # 提取文件名（不含扩展名）
    name_without_ext = os.path.splitext(filename)[0]
    
    # 分割数字和非数字部分
    parts = re.split(r'(\d+)', name_without_ext)
    
    # 将数字部分转换为整数，非数字部分保持字符串
    converted = []
    for part in parts:
        if part.isdigit():
            converted.append(int(part))
        else:
            converted.append(part)
    
    return converted

def reorder_interlaced_files(image_files):
    """
    重新排序隔行扫描的文件（之字形扫描模式）
    
    用户最新反馈：改为 奇数倒序 + 偶数正序
    这样可以避免在图像处理时翻转D轴
    
    输入: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]
    输出: [9, 7, 5, 3, 1, 0, 2, 4, 6, 8] (奇数倒序 + 偶数正序)
    
    示例（大数据集360张）:
    输入: [0, 1, 2, ..., 179, 180, 181, ..., 358, 359]
    输出: [359, 357, ..., 3, 1, 0, 2, 4, ..., 358]
    """
    even_files = [f for i, f in enumerate(image_files) if i % 2 == 0]  # 偶数索引
    odd_files = [f for i, f in enumerate(image_files) if i % 2 == 1]   # 奇数索引
    
    odd_files_reversed = list(reversed(odd_files))  # 奇数倒序
    # 偶数保持正序
    
    return odd_files_reversed + even_files

def detect_view_orientation(folder_name):
    """
    检测扫描视角
    
    返回:
        'axial', 'coronal', 'sagittal' 或 'unknown'
    """
    folder_lower = folder_name.lower()
    
    if 'axial' in folder_lower:
        return 'axial'
    elif 'coronal' in folder_lower:
        return 'coronal'
    elif 'sagittal' in folder_lower:
        return 'sagittal'
    else:
        return 'unknown'