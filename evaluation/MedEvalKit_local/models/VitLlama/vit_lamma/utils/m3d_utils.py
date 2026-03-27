import os
import re
from collections import Counter
from typing import List, Union

import cv2
import numpy as np
from PIL import Image


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

def load_m3d_image_folder(
    image_path: Union[str, List[str], Image.Image, List[Image.Image]],
) -> tuple:
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]
    if not image_files:
        raise ValueError(f"No image files found in directory: {image_path}")

    # sort and reorder
    image_files.sort(key=_natural_key)
    image_files = reorder_interlaced_files(image_files)

    images_3d = load_image_slices_strict(image_path, image_files)

    metadata = {
        'spacing': (1.0, 1.0, 1.0),
        'shape': images_3d.shape,
        'dtype': str(images_3d.dtype),
    }

    images_3d = np.transpose(images_3d, (2, 0, 1))
    return images_3d, metadata


def validate_and_filter_slices(images_3d, min_slices=1):
    if len(images_3d) < min_slices:
        return None

    # 只比较空间维度 (H, W)
    spatial_shapes = [img.shape[:2] for img in images_3d]
    shape_counts = Counter(spatial_shapes)
    most_common_hw, _ = shape_counts.most_common(1)[0]  # (H, W)

    filtered_images = []
    target_h, target_w = most_common_hw

    for img in images_3d:
        current_hw = img.shape[:2]
        if current_hw != most_common_hw:
            resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            filtered_images.append(resized)
        else:
            filtered_images.append(img)

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


def _natural_key(filename):
    """
    自然排序key函数
    正确处理: 1.png, 2.png, ..., 10.png, 11.png
    """
    name_without_ext = os.path.splitext(filename)[0]
    parts = re.split(r'(\d+)', name_without_ext)
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
    奇数倒序 + 偶数正序
    """
    even_files = [f for i, f in enumerate(image_files) if i % 2 == 0]
    odd_files = [f for i, f in enumerate(image_files) if i % 2 == 1]
    odd_files_reversed = list(reversed(odd_files))
    return odd_files_reversed + even_files


def detect_view_orientation(folder_name):
    """
    检测扫描视角
    返回: 'axial', 'coronal', 'sagittal' 或 'unknown'
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
