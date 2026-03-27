#!/usr/bin/env python3
"""
Merge bridge-only finetuned checkpoints into a single Hugging Face folder.

Example:
python3 LaMed/src/utils/merge_bridge_weights_and_save_hf_model.py \
  --model_name_or_path "./LaMed/pretrained_model/huggingface_models/Meta-Llama-3.1-8B-Instruct" \
  --model_with_bridge "./LaMed/output/bridge_only/LaMed-Llama3.1-8B-finetune-MAE-CTRATE-no-compress_checkpoint-96000_v3_bridge_only/checkpoint-8000"
"""

import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

import torch
import transformers
import numpy as np
from transformers import AutoTokenizer

# project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from LaMed.src.model.language_model import (  # noqa: E402
    LamedLlamaForCausalLM,
    LamedQwenForCausalLM,
)
from LaMed.src.train.vision_resampler_bridge import (  # noqa: E402
    DirectConnection,
    ResamplerBridge,
)


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(
        default="./LaMed/pretrained_model/huggingface_models/Meta-Llama-3.1-8B-Instruct",
        metadata={"help": "Base LLM path before bridge-only finetuning."},
    )
    model_type: Optional[str] = field(default="llama3", metadata={"help": "llama2, llama3, phi3, qwen"})
    model_with_bridge: Optional[str] = field(
        default="./LaMed/output/bridge_only/LaMed-Llama3.1-8B-finetune-MAE-CTRATE-no-compress_checkpoint-96000_v3_bridge_only/checkpoint-8000",
        metadata={"help": "Path to finetuned checkpoint (directory or file)."},
    )

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(default=False, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."})

    # image
    image_channel: int = field(default=1)
    img_size: tuple = field(default=(128, 256, 256))
    patch_size: tuple = field(default=(8, 16, 16))

    # vision
    vision_tower: Optional[str] = field(default="vit3d")  # None, "vit3d" "visd"
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default=None, metadata={"help": "Path to pretrained model for ViT."})
    freeze_vision_tower: bool = field(default=False)
    bridge_mode: Optional[str] = field(
        default="resampler_bridge", metadata={"help": "resampler_bridge or direct"}
    )
    resampler_num_tokens: Optional[int] = field(
        default=512, metadata={"help": "K tokens emitted by resampler when bridge_mode=resampler_bridge"}
    )
    bridge_mlp_ratio: float = field(
        default=2.0, metadata={"help": "MLP ratio inside bridge when bridge_mode=resampler_bridge"}
    )

    # projector
    mm_projector_type: Optional[str] = field(default="spp")
    proj_layer_type: str = field(default="mlp", metadata={"help": "Type of projector in Perceiver. options: [linear, mlp]."})
    proj_layer_num: int = field(default=2, metadata={"help": "Number of projectors in Perceiver."})
    proj_pooling_type: str = field(default="spatial", metadata={"help": "Type of pooling in Perceiver. options: [spatial, sequence]."})
    proj_pooling_size: int = field(default=2, metadata={"help": "Size of pooling in Perceiver."})

    # segvol
    segmentation_module: Optional[str] = field(default=None, metadata={"help": "segvol"})
    pretrain_seg_module: str = field(default=None, metadata={"help": "Pretrained segvol model."})


@dataclass
class BridgeTrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default="")


def resolve_checkpoint_paths(model_with_bridge_path: str) -> Tuple[str, str, bool]:
    """Normalize checkpoint paths and tell whether the input was a directory."""
    resolved_path = os.path.abspath(model_with_bridge_path)
    is_dir = os.path.isdir(resolved_path)
    checkpoint_dir = resolved_path if is_dir else os.path.dirname(resolved_path)
    return resolved_path, checkpoint_dir, is_dir


def apply_checkpoint_config(model_args: ModelArguments, config_dict: Optional[Dict]):
    """Copy relevant fields saved during finetuning into current model_args."""
    if not config_dict:
        return

    key_map = {
        "image_channel": "image_channel",
        "img_size": "img_size",
        "patch_size": "patch_size",
        "vision_tower": "vision_tower",
        "vision_select_layer": "vision_select_layer",
        "vision_select_feature": "vision_select_feature",
        "mm_projector_type": "mm_projector_type",
        "proj_layer_type": "proj_layer_type",
        "proj_layer_num": "proj_layer_num",
        "proj_pooling_type": "proj_pooling_type",
        "proj_pooling_size": "proj_pooling_size",
        "segmentation_module": "segmentation_module",
        "bridge_mode": "bridge_mode",
        "resampler_num_tokens": "resampler_num_tokens",
        "bridge_mlp_ratio": "bridge_mlp_ratio",
    }

    for ckpt_key, arg_key in key_map.items():
        if ckpt_key not in config_dict:
            continue
        value = config_dict[ckpt_key]
        if arg_key in ("img_size", "patch_size") and isinstance(value, list):
            value = tuple(value)
        setattr(model_args, arg_key, value)
        print(f"[CONFIG] Loaded {arg_key}={value} from checkpoint config")


def load_qwen_text_config(model_path: str):
    """
    Extract the text-only config from a Qwen3-VL style config.json.
    Falls back to the root config fields when text_config is absent.
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        raw_cfg = json.load(f)
    text_cfg = raw_cfg.get("text_config", raw_cfg)
    text_cfg = dict(text_cfg)
    for key in ("vocab_size", "bos_token_id", "eos_token_id"):
        if key not in text_cfg and key in raw_cfg:
            text_cfg[key] = raw_cfg[key]
    text_cfg.setdefault("model_type", "lamed_qwen")
    return LamedQwenForCausalLM.config_class.from_dict(text_cfg)


def normalize_projector_key(k: str) -> str:
    new_k = k.replace("mm_projector.proj", "mm_projector.projector")
    if new_k.endswith(".mm_projector.projector.weight"):
        new_k = new_k.replace(".projector.weight", ".projector.0.weight")
    if new_k.endswith(".mm_projector.projector.bias"):
        new_k = new_k.replace(".projector.bias", ".projector.0.bias")
    return new_k


def find_weight_file(checkpoint_dir: str, provided_path: Optional[str], is_dir: bool) -> str:
    if not is_dir and os.path.isfile(provided_path):
        return provided_path
    candidates = [
        "pytorch_model.bin",
        "model.safetensors",
        "consolidated.00.pth",
        "model_with_lora.bin",  # fallback naming
        "adapter_model.bin",
    ]
    for cand in candidates:
        path = os.path.join(checkpoint_dir, cand)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No model weights found under {checkpoint_dir} (checked {candidates})")


def build_bridge_projector(model, model_args):
    """Recreate the bridge module used during training (resampler or direct)."""
    if model_args.vision_tower is None:
        return None

    vision_hidden_size = model.get_model().vision_tower.hidden_size
    llm_hidden_size = model.config.hidden_size

    if model_args.vision_tower and "visd" in model_args.vision_tower.lower():
        num_patches = getattr(model.get_model().vision_tower, "num_patches", None)
        if num_patches is None:
            img_size = model_args.img_size
            patch_size = model_args.patch_size
            num_patches = int(np.prod([img_size[i] // patch_size[i] for i in range(3)]))
    else:
        img_size = model_args.img_size
        patch_size = model_args.patch_size
        num_patches = int(np.prod([img_size[i] // patch_size[i] for i in range(3)]))

    use_resampler = (model_args.bridge_mode or "").lower() == "resampler_bridge"
    if use_resampler:
        k_tokens = model_args.resampler_num_tokens
        projector = ResamplerBridge(
            d_enc=vision_hidden_size,
            d_llm=llm_hidden_size,
            k=k_tokens,
            num_heads=8,
            mlp_ratio=model_args.bridge_mlp_ratio,
        )
        print(f"[Bridge] Using ResamplerBridge with k={k_tokens}, mlp_ratio={model_args.bridge_mlp_ratio}")
    else:
        projector = DirectConnection(
            in_dim=vision_hidden_size,
            out_dim=llm_hidden_size,
            num_tokens=num_patches,
        )
        print(f"[Bridge] Using DirectConnection with {num_patches} tokens")

    return projector


def main():
    parser = transformers.HfArgumentParser((ModelArguments, BridgeTrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    bridge_spec_path, checkpoint_dir, checkpoint_is_dir = resolve_checkpoint_paths(model_args.model_with_bridge)
    if not os.path.exists(bridge_spec_path):
        raise FileNotFoundError(f"--model_with_bridge path does not exist: {model_args.model_with_bridge}")

    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            ckpt_config = json.load(f)
        apply_checkpoint_config(model_args, ckpt_config)
    else:
        print(f"[WARN] config.json not found at {config_path}; using default model args.")

    print("Tokenizer preparation")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
    )
    special_token = {"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]}
    tokenizer.add_special_tokens(special_token)
    tokenizer.add_tokens("[SEG]")

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if "llama3" in model_args.model_type:
        tokenizer.pad_token = tokenizer.eos_token
    elif "qwen" in model_args.model_type:
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    model_args.vocab_size = len(tokenizer)
    print("seg_token_id: ", model_args.seg_token_id)
    print("vocab_size: ", model_args.vocab_size)

    print("Model preparation")
    qwen_text_config = None
    if model_args.model_type and "qwen" in model_args.model_type:
        qwen_text_config = load_qwen_text_config(model_args.model_name_or_path)

    if "llama" in model_args.model_type:
        model = LamedLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    elif "phi3" in model_args.model_type:
        from LaMed.src.model.language_model import LamedPhi3ForCausalLM  # lazy import

        model = LamedPhi3ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    elif "qwen" in model_args.model_type:
        model = LamedQwenForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            config=qwen_text_config,
            ignore_mismatched_sizes=True,
        )
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")

    model.config.seg_token_id = model_args.seg_token_id
    model.config.bridge_mode = model_args.bridge_mode
    model.config.resampler_num_tokens = model_args.resampler_num_tokens
    model.config.bridge_mlp_ratio = model_args.bridge_mlp_ratio

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)
        projector = build_bridge_projector(model, model_args)
        if projector is not None:
            model.get_model().mm_projector = projector
    if model_args.segmentation_module is not None:
        model.get_model().initialize_seg_modules(model_args=model_args)

    model_args.num_new_tokens = 4
    model.initialize_vision_tokenizer(model_args, tokenizer)

    print("Load bridge weights")
    weight_path = find_weight_file(checkpoint_dir, bridge_spec_path, checkpoint_is_dir)
    state_dict_raw = torch.load(weight_path, map_location="cpu")

    # Optionally override mm_projector / embed_tokens from separate shard if present
    mm_proj_path = os.path.join(checkpoint_dir, "mm_projector.bin")
    if os.path.isfile(mm_proj_path):
        mm_state = torch.load(mm_proj_path, map_location="cpu")
        state_dict_raw.update(mm_state)

    # Normalize projector naming for compatibility
    state_dict = {}
    for k, v in state_dict_raw.items():
        state_dict[normalize_projector_key(k)] = v

    if not training_args.output_dir:
        run_root = os.path.dirname(os.path.normpath(checkpoint_dir))
        checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))
        training_args.output_dir = os.path.join(run_root, "hf_merged", checkpoint_name)
        print(f"output directory: {training_args.output_dir}")

    base_state_dict = model.state_dict()
    base_state_dict.update(state_dict)
    model.load_state_dict(base_state_dict, strict=True)

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    torch.save(model.state_dict(), os.path.join(training_args.output_dir, "merged_model.bin"))

    model.model.config.architectures = model.__class__.__name__
    model._name_or_path = training_args.output_dir

    print("Save pretrained")
    model.config.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print("Saved model to: ", training_args.output_dir)
    print("Finish")


if __name__ == "__main__":
    main()

