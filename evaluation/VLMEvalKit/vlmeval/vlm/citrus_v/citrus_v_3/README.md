# CitrusV 3.0 - Medical Multimodal AI based on Qwen3-VL

CitrusV 3.0 extends Qwen3-VL with medical AI capabilities while maintaining full compatibility with Qwen3-VL pretrained weights.

## 🌟 Features

### Core Capabilities
- ✅ **Full Qwen3-VL Compatibility**: Load and use any Qwen3-VL checkpoint directly
- ✅ **NIfTI Medical Image Support**: Native support for `.nii` and `.nii.gz` formats
- ✅ **Video Compression**: Efficient compression at patch embedding level (HuluMed-style)
- ✅ **3D Medical Imaging**: Process volumetric medical scans
- ✅ **Multi-Modal**: Images, videos, NIfTI, and text

### Technical Highlights
- **Patch-Level Compression**: More efficient than frame-level compression
- **Semantic Awareness**: Compression based on visual feature differences
- **Backward Compatible**: Works with all Qwen3-VL features and weights
- **Medical Optimized**: Designed for medical imaging workflows

## 📦 Installation

```bash
# Required packages
pip install transformers>=4.57.0
pip install torch>=2.0.0
pip install nibabel  # For NIfTI support
pip install decord   # For video processing

# Optional for specific formats
pip install opencv-python
pip install imageio
```

## 🚀 Quick Start

### Volume Inference (NIfTI) via Chat Template

```python
import torch
from transformers import AutoTokenizer
from architectures.citrus_v_3 import (
    CitrusV3ForConditionalGeneration,
    CitrusV3Processor,
)

model_path = "/mnt/workspace/offline/caoxuyang5/code/ms-swift-390/output/v110-20251029-153218/checkpoint-6602"
model = CitrusV3ForConditionalGeneration.from_pretrained(
    model_path, 
    dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2", 
    device_map='auto'
)
processor = CitrusV3Processor.from_pretrained(model_path)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_3d",
                "image_3d": "/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-CAP/M3D_Cap_nii/ct_quizze/005329/Axial_C__portal_venous_phase.nii.gz",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)


generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

### Processing Videos with Compression

```python
# Video compression happens automatically at model level
inputs = processor(
    text="Describe what happens in this surgical video",
    videos="path/to/surgery.mp4"
)

outputs = model.generate(**inputs, max_new_tokens=512)
```

## 🏗️ Architecture

### Components

```
CitrusV 3.0
├── Configuration (CitrusV3Config)
│   ├── Extends Qwen3VLConfig
│   └── Adds: NIfTI support, video compression params
│
├── Model (CitrusV3ForConditionalGeneration)
│   ├── Vision Encoder (from Qwen3-VL)
│   ├── Language Model (from Qwen3-VL)
│   └── Video Compression (patch-level)
│
└── Processors
    ├── Image Processor (with NIfTI)
    ├── Video Processor
    └── Unified Processor
```

### Key Differences from Qwen3-VL

| Feature | Qwen3-VL | CitrusV 3.0 |
|---------|----------|-------------|
| Standard Images | ✅ | ✅ |
| Videos | ✅ | ✅ |
| NIfTI Support | ❌ | ✅ |
| Video Compression | ❌ | ✅ (Patch-level) |
| 3D Medical Imaging | ❌ | ✅ |
| Pretrained Weights | ✅ | ✅ (Compatible) |

## 📖 Usage Examples

### Example 1: Medical Image Analysis

```python
# Analyze CT scan
inputs = processor(
    text="Identify any pathological findings in this scan",
    images="patient_ct.nii.gz"
)

outputs = model.generate(**inputs)
diagnosis = processor.batch_decode(outputs)[0]
```

### Example 2: Multi-Image Comparison

```python
# Compare multiple scans
inputs = processor(
    text="Compare these two scans and describe changes",
    images=["scan_t1.nii.gz", "scan_t2.nii.gz"]
)

outputs = model.generate(**inputs)
```

### Example 3: Video Understanding

```python
# Surgical video analysis
inputs = processor(
    text="Describe the surgical procedure step by step",
    videos="surgery_video.mp4"
)

outputs = model.generate(**inputs)
```

## ⚙️ Configuration

### Video Compression Settings

```python
from architectures.citrus_v_3 import CitrusV3Config

config = CitrusV3Config.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

# Enable/disable compression
config.video_compression_enabled = True

# Compression threshold (0.0-1.0)
# Higher = more compression, lower = better quality
config.video_compression_threshold = 0.1

# Minimum tokens per frame
config.video_min_tokens_per_frame = 1
```

### NIfTI Processing

```python
# NIfTI support is automatic
# Files with .nii or .nii.gz are automatically detected
inputs = processor(images="scan.nii.gz")
```

## 🔬 Technical Details

### Video Compression Mechanism

CitrusV 3.0 implements compression at the **patch embedding level**:

1. **Vision Encoding**: Process video frames through vision encoder
2. **Feature Extraction**: Extract patch embeddings for each frame
3. **Difference Computation**: Calculate inter-frame differences of embeddings
4. **Compression**: Keep frames with significant changes
5. **Token Reduction**: Reduce total visual tokens while preserving semantics

Benefits:
- More semantic than pixel-level compression
- Reduces computational cost
- Maintains visual understanding quality

### NIfTI Support

NIfTI files are automatically:
1. Loaded using `nibabel`
2. Converted to 2D slices
3. Processed as image sequences
4. Fed to the vision encoder

## 📊 Comparison with CitrusV 2.5

| Feature | CitrusV 2.5 | CitrusV 3.0 |
|---------|-------------|-------------|
| Base Model | Qwen2.5-VL | **Qwen3-VL** |
| Vision Encoder | Qwen2.5 Vision | **Qwen3 Vision** |
| Text Model | Qwen2.5 | **Qwen3** |
| NIfTI Support | ✅ | ✅ |
| Video Compression | ✅ | ✅ |
| Performance | Good | **Better** |

## 🧪 Testing

```bash
# Test imports
python -c "from architectures.citrus_v_3 import CitrusV3ForConditionalGeneration; print('✅ Import successful')"

# Test with Qwen3-VL weights
python architectures/citrus_v_3/test_inference.py
```

## 📚 API Reference

### CitrusV3Config

```python
CitrusV3Config(
    support_nifti=True,
    video_compression_enabled=True,
    video_compression_threshold=0.1,
    video_min_tokens_per_frame=1,
    **qwen3_vl_config_params
)
```

### CitrusV3ForConditionalGeneration

```python
model = CitrusV3ForConditionalGeneration.from_pretrained(model_path)
outputs = model.generate(**inputs, max_new_tokens=512)
```

### CitrusV3Processor

```python
processor = CitrusV3Processor(
    image_processor=image_processor,
    tokenizer=tokenizer,
    video_processor=video_processor
)

inputs = processor(text=prompt, images=images, videos=videos)
```

## 🤝 Integration with Swift Framework

CitrusV 3.0 can be registered in Swift for training:

```bash
# Register model
swift sft \
    --model_type citrus_v_3 \
    --model /path/to/Qwen3-VL-4B-Instruct \
    --dataset your_dataset \
    --output_dir output/citrus_v_3
```

## 📝 License

Same as Qwen3-VL (Apache 2.0)

## 🙏 Acknowledgments

- **Qwen Team**: For the excellent Qwen3-VL foundation
- **HuluMed Team**: For the video compression inspiration
- **Transformers**: For the infrastructure

## 📮 Contact

For questions or issues, please refer to the main CitrusV documentation.

---

**Status**: ✅ Ready for Use

**Compatibility**: Qwen3-VL 4B/72B Instruct models

**Version**: 3.0.0

