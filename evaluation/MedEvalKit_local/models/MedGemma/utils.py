import os
import base64
import PIL.Image
import io

import numpy as np


# def norm(ct_vol: np.ndarray, min: float, max: float) -> np.ndarray:
#   """Window and normalize CT imaging Houndsfield values to values 0 - 255."""
#   ct_vol = np.clip(ct_vol, min, max)  # Clip the imaging value range
#   ct_vol = ct_vol.astype(np.float32)
    
#   ct_vol -= min
#   ct_vol /= (max - min) # Norm to values between 0 - 1.0
#   ct_vol *= 255.0  # Norm to values been 0 - 255.0
#   return ct_vol


# def window(ct_vol: np.ndarray, dcm: pydicom.Dataset) -> np.ndarray:
#   # Window CT slice imaging with three windows (wide, mediastinum(chest), brain)
#   # Imaging will appear color when visualized, RGB channels contain different
#   # representations of the data.
#   window_clips = [(-1024, 1024), (-135, 215), (0, 80)]
#   return np.stack([norm(ct_vol, clip[0], clip[1]) for clip in window_clips], axis=-1)



def _encode(data: np.ndarray) -> str:
  """Encode CT slice imaging inline in prompt."""
  # Image format to encode ct slice images as.
  # options: "jpeg" or "png"
  format = "jpeg"
  with io.BytesIO() as img_bytes:
    with PIL.Image.fromarray(data) as img:
      img.save(img_bytes, format=format)
    img_bytes.seek(0)
    encoded_string = base64.b64encode(img_bytes.getbuffer()).decode("utf-8")
  return f"data:image/{format};base64,{encoded_string}"


def encode(image, format="PNG"):
    """Encode CT slice imaging inline in prompt."""
    # Image format to encode ct slice images as.
    # options: "jpeg" or "png"
    # 1. 创建一个内存缓冲区
    buffered = io.BytesIO()
    # 2. 将 PIL 图像保存到缓冲区（需要指定格式，如 PNG 或 JPEG）
    image.save(buffered, format=format)
    # 3. 将缓冲区内容转换为 bytes，并进行 base64 编码
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # 4. 可选：添加 Data URI 前缀，使其能直接在 HTML/浏览器中使用
    return f"data:image/{format.lower()};base64,{img_str}"
