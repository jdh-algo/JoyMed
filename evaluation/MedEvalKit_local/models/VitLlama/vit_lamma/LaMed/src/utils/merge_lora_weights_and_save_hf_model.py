# example run: 
# Example usage:
# python3 LaMed/src/utils/merge_lora_weights_and_save_hf_model.py \
#   --model_name_or_path "./LaMed/pretrained_model/huggingface_models/Meta-Llama-3.1-8B-Instruct" \
#   --model_with_lora "./LaMed/output/LaMed-Llama3.1-8B-finetune-MAE-CTRATE-no-compress/checkpoint-3000/"

# python3 LaMed/src/utils/merge_lora_weights_and_save_hf_model.py \
#   --model_name_or_path "./LaMed/pretrained_model/huggingface_models/Qwen3-VL-8B-Instruct" \
#   --model_with_lora "./LaMed/output/LaMed-Qwen3-VL-8B-finetune-MAE_CTRATE_96000-no-compress_checkpoint-96000_v2/checkpoint-7000/" \
#   --model_type "qwen"

# This script merges the weights of a language model (e.g., Llama) that has been finetuned using LoRA adapters,
# with or without 3D vision model adapters (e.g., ViT, MAE), and saves the resulting merged model in Huggingface format.
# The script loads model and tokenizer, manages special and segmentation tokens, 
# applies or merges LoRA adapters if enabled, reconstructs visual tower/projector modules as needed,
# and saves the merged model state dict, config, and tokenizer for later Huggingface-based inference or finetuning.

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import json
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import transformers
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedQwenForCausalLM
from LaMed.src.train.vision_resampler_bridge import DirectConnection, ResamplerBridge

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(
        default="./LaMed/pretrained_model/huggingface_models/Meta-Llama-3.1-8B-Instruct",
        metadata={"help": "Base LLM path before LoRA finetuning."})
    model_type: Optional[str] = field(default="llama3", metadata={"help": "llama2, llama3, phi3, qwen"})

    model_with_lora: Optional[str] = field(
        default="./LaMed/output/LaMed-Llama3.1-8B-finetune-MAE-CTRATE-no-compress/checkpoint-1000/",
        metadata={"help": "Path to model_with_lora.bin or its containing checkpoint directory."})

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(default=False, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."})

    # image
    image_channel: int = field(default=1)
    img_size: tuple = field(default=(128, 256, 256))
    patch_size: tuple = field(default=(8, 16, 16))

    # vision
    vision_tower: Optional[str] = field(default="vit3d") # None, "vit3d" "visd"
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default=None, metadata={"help": "Path to pretrained model for ViT."})
    freeze_vision_tower: bool = field(default=False)
    bridge_mode: Optional[str] = field(default="resampler_bridge", metadata={"help": "resampler_bridge or direct"})
    resampler_num_tokens: Optional[int] = field(default=512, metadata={"help": "K tokens emitted by resampler when bridge_mode=resampler_bridge"})
    bridge_mlp_ratio: float = field(default=2.0, metadata={"help": "MLP ratio inside bridge when bridge_mode=resampler_bridge"})

    # projector
    mm_projector_type: Optional[str] = field(default='spp')
    proj_layer_type: str = field(default="mlp", metadata={"help": "Type of projector in Perceiver. options: [linear, mlp]."})
    proj_layer_num: int = field(default=2, metadata={"help": "Number of projectors in Perceiver."})
    proj_pooling_type: str = field(default="spatial", metadata={"help": "Type of pooling in Perceiver. options: [spatial, sequence]."})
    proj_pooling_size: int = field(default=2, metadata={"help": "Size of pooling in Perceiver."})

    # segvol
    segmentation_module: Optional[str] = field(default=None, metadata={"help": "segvol"})
    pretrain_seg_module: str = field(default=None, metadata={"help": "Pretrained segvol model."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    lora_enable: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    output_dir: str = ""


def resolve_checkpoint_paths(model_with_lora_path: str):
    """Normalize checkpoint paths and tell whether the input was a directory."""
    resolved_path = os.path.abspath(model_with_lora_path)
    is_dir = os.path.isdir(resolved_path)
    checkpoint_dir = resolved_path if is_dir else os.path.dirname(resolved_path)
    return resolved_path, checkpoint_dir, is_dir


def apply_checkpoint_config(model_args, config_dict):
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


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)


def has_direct_projector_weights(state_dict):
    return any(".mm_projector.proj" in k for k in state_dict.keys())


def normalize_projector_key(k: str, bridge_mode: Optional[str]) -> str:
    """
    Normalize projector naming for older checkpoints.
    Avoid renaming when bridge_mode is explicit direct (DirectConnection uses .proj).
    """
    if bridge_mode and bridge_mode.lower() == "direct":
        return k
    new_k = k.replace("mm_projector.proj", "mm_projector.projector")
    if new_k.endswith(".mm_projector.projector.weight"):
        new_k = new_k.replace(".projector.weight", ".projector.0.weight")
    if new_k.endswith(".mm_projector.projector.bias"):
        new_k = new_k.replace(".projector.bias", ".projector.0.bias")
    return new_k


def find_weight_file(checkpoint_dir: str, provided_path: str, is_dir: bool) -> str:
    if not is_dir and os.path.isfile(provided_path):
        return provided_path
    candidates = [
        "model_with_lora.bin",
        "pytorch_model.bin",
        "adapter_model.bin",
        "model.safetensors",
        "consolidated.00.pth",
    ]
    for cand in candidates:
        cand_path = os.path.join(checkpoint_dir, cand)
        if os.path.isfile(cand_path):
            return cand_path
    raise FileNotFoundError(f"Directory {checkpoint_dir} does not contain any of {candidates}")


def build_bridge_projector(model, model_args):
    """
    Recreate the bridge module (resampler or direct) used during training.
    This ensures the shape matches saved weights before we load them.
    """
    if model_args.vision_tower is None:
        return None

    vision_tower = model.get_model().vision_tower
    vision_hidden_size = vision_tower.hidden_size
    llm_hidden_size = model.config.hidden_size

    is_visd = False
    vision_tower_name = str(getattr(model_args, "vision_tower", "")).lower()
    model_with_lora_path = str(getattr(model_args, "model_with_lora", "")).lower()
    if "visd" in vision_tower_name or "visd" in model_with_lora_path:
        is_visd = True

    num_patches = getattr(vision_tower, "num_patches", None)
    if num_patches is None:
        img_size = tuple(model_args.img_size)
        patch_size = tuple(model_args.patch_size)
        num_patches = int(np.prod([img_size[i] // patch_size[i] for i in range(3)]))

    if is_visd:
        visd_patch_stride = getattr(vision_tower, "patch_stride", None)
        if visd_patch_stride is not None:
            model.config.patch_size = tuple(visd_patch_stride)
            model_args.patch_size = tuple(visd_patch_stride)

    bridge_mode = (model_args.bridge_mode or "resampler_bridge").lower()
    if bridge_mode == "direct":
        projector = DirectConnection(in_dim=vision_hidden_size, out_dim=llm_hidden_size, num_tokens=num_patches)
        print(f"[Bridge] Using DirectConnection with {num_patches} tokens")
    else:
        k_tokens = model_args.resampler_num_tokens
        projector = ResamplerBridge(
            d_enc=vision_hidden_size,
            d_llm=llm_hidden_size,
            k=k_tokens,
            num_heads=8,
            mlp_ratio=model_args.bridge_mlp_ratio,
        )
        print(f"[Bridge] Using ResamplerBridge with k={k_tokens}, mlp_ratio={model_args.bridge_mlp_ratio}")

    model.config.bridge_mode = model_args.bridge_mode
    model.config.resampler_num_tokens = model_args.resampler_num_tokens
    model.config.bridge_mlp_ratio = model_args.bridge_mlp_ratio
    model.config.num_vision_tokens = getattr(projector, "proj_out_num", None)
    return projector


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
    # keep critical ids if present in the root config
    for key in ("vocab_size", "bos_token_id", "eos_token_id"):
        if key not in text_cfg and key in raw_cfg:
            text_cfg[key] = raw_cfg[key]
    text_cfg.setdefault("model_type", "lamed_qwen")
    return LamedQwenForCausalLM.config_class.from_dict(text_cfg)


def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    lora_spec_path, checkpoint_dir, checkpoint_is_dir = resolve_checkpoint_paths(model_args.model_with_lora)
    if not os.path.exists(lora_spec_path):
        raise FileNotFoundError(f"--model_with_lora path does not exist: {model_args.model_with_lora}")

    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            ckpt_config = json.load(f)
        apply_checkpoint_config(model_args, ckpt_config)
    else:
        print(f"[WARN] config.json not found at {config_path}; using default model args.")

    # Load saved adapter/bridge weights before building model to detect bridge type.
    lora_path = find_weight_file(checkpoint_dir, lora_spec_path, checkpoint_is_dir)
    raw_state_dict = torch.load(lora_path, map_location="cpu")

    # mm_projector / embed_tokens saved separately by LaMedTrainer
    mm_proj_path = os.path.join(checkpoint_dir, "mm_projector.bin")
    if os.path.isfile(mm_proj_path):
        mm_state = torch.load(mm_proj_path, map_location="cpu")
        raw_state_dict.update(mm_state)
        print(f"[INFO] Loaded additional projector/embed weights from {mm_proj_path}")

    # Auto-detect direct bridge when checkpoint uses DirectConnection naming
    if has_direct_projector_weights(raw_state_dict):
        model_args.bridge_mode = "direct"

    print("Tokenizer preparation")
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
    )

    # Define and add special tokens
    special_token = {"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]}
    tokenizer.add_special_tokens(
        special_token
    )
    tokenizer.add_tokens("[SEG]")

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if 'llama3' in model_args.model_type:
        # Use tokenizer's eos_token_id (defaults to 128009 for Llama 3.1)
        # Set pad_token to eos_token (standard for Llama 3)
        tokenizer.pad_token = tokenizer.eos_token
    elif 'qwen' in model_args.model_type:
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    model_args.vocab_size = len(tokenizer)
    print("seg_token_id: ", model_args.seg_token_id)
    print("vocab_size: ", model_args.vocab_size)

    print("Model preparation")
    qwen_text_config = None
    if model_args.model_type and 'qwen' in model_args.model_type:
        qwen_text_config = load_qwen_text_config(model_args.model_name_or_path)
    
    if 'llama' in model_args.model_type:
        model = LamedLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )
    elif 'phi3' in model_args.model_type:
        from LaMed.src.model.language_model import LamedPhi3ForCausalLM
        model = LamedPhi3ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )
    elif 'qwen' in model_args.model_type:
        model = LamedQwenForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            config=qwen_text_config,
            ignore_mismatched_sizes=True
        )
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")

    model.config.seg_token_id = model_args.seg_token_id

    # initialize vision and seg modules on LLM
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)
        projector = build_bridge_projector(model, model_args)
        if projector is not None:
            model.get_model().mm_projector = projector
    if model_args.segmentation_module is not None:
        model.get_model().initialize_seg_modules(model_args=model_args)

    model_args.num_new_tokens = 4
    model.initialize_vision_tokenizer(model_args, tokenizer)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    print("Load weights with LoRA")
    lora_dir = checkpoint_dir
    state_dict = {}
    for k, v in raw_state_dict.items():
        new_k = normalize_projector_key(k, model_args.bridge_mode)
        state_dict[new_k] = v

    if not training_args.output_dir:
        run_root = os.path.dirname(os.path.normpath(lora_dir))  # .../LaMed/output/...-no-compress
        checkpoint_name = os.path.basename(os.path.normpath(lora_dir))
        training_args.output_dir = os.path.join(run_root, "hf_merged", checkpoint_name)
        print(f"output directory: {training_args.output_dir}")

    base_state_dict = model.state_dict()
    for k, v in state_dict.items():
        base_state_dict[k] = v
    model.load_state_dict(base_state_dict, strict=True)

    print("Merge weights with LoRA")
    model = model.merge_and_unload()
    state_dict = model.state_dict()

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    torch.save(state_dict, os.path.join(training_args.output_dir, 'merged_model.bin'))

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
