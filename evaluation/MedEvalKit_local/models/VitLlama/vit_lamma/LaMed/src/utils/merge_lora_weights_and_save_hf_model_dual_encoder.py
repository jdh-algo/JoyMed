# example run: 
# python3 LaMed/src/utils/merge_lora_weights_and_save_hf_model_dual_encoder.py \
#   --model_name_or_path "./LaMed/pretrained_model/huggingface_models/Meta-Llama-3.1-8B-Instruct" \
#   --model_with_lora "./LaMed/output/LaMed-Llama3.1-8B-finetune-MAE_CLIP_CTRATE-no-compress_checkpoint-96000_v2/checkpoint-7000/"

# This script merges weights from a LoRA-finetuned dual-encoder multimodal LLM and saves the result in Huggingface format.
# It loads both the language model and the vision encoders (e.g., MAE and CLIP), applies LoRA updates,
# and ensures tokenizer and special token alignment.
# The script handles projector module reconstruction for dual visual towers, manages special tokens
# (like segmentation, image patch), and supports restoration from LoRA adapter-only or full checkpoints.
# At the end, it saves the merged model, tokenizer, and config for Huggingface-compatible inference or further fine-tuning.

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
from LaMed.src.model.language_model import LamedLlamaForCausalLM
from LaMed.src.model.multimodal_encoder.builder import build_vision_tower

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(
        default="./LaMed/pretrained_model/huggingface_models/Meta-Llama-3.1-8B-Instruct",
        metadata={"help": "Base LLM path before LoRA finetuning."})
    model_type: Optional[str] = field(default="llama3", metadata={"help": "llama2, llama3, phi3"})

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
    dual_vision: bool = field(default=False, metadata={"help": "Use dual MAE+CLIP encoders"})
    vision_tower: Optional[str] = field(default="vit3d") # None, "vit3d"
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default=None, metadata={"help": "Path to pretrained model for ViT."})
    pretrain_vision_model_mae: Optional[str] = field(default=None, metadata={"help": "Path to MAE vision encoder."})
    pretrain_vision_model_clip: Optional[str] = field(default=None, metadata={"help": "Path to CLIP vision encoder."})
    freeze_vision_tower: bool = field(default=False)

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
        "dual_vision": "dual_vision",
        "vision_select_layer": "vision_select_layer",
        "vision_select_feature": "vision_select_feature",
        "mm_projector_type": "mm_projector_type",
        "proj_layer_type": "proj_layer_type",
        "proj_layer_num": "proj_layer_num",
        "proj_pooling_type": "proj_pooling_type",
        "proj_pooling_size": "proj_pooling_size",
        "segmentation_module": "segmentation_module",
        "pretrain_vision_model_mae": "pretrain_vision_model_mae",
        "pretrain_vision_model_clip": "pretrain_vision_model_clip",
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


def install_direct_mm_projector(model, model_args):
    vision_hidden_size = model.get_model().vision_tower.hidden_size
    llm_hidden_size = model.config.hidden_size
    img_size = tuple(model_args.img_size)
    patch_size = tuple(model_args.patch_size)
    per_dim = [max(1, img // p) for img, p in zip(img_size, patch_size)]
    num_tokens = int(np.prod(per_dim))

    class DirectConnection(nn.Module):
        def __init__(self, in_dim, out_dim, num_tokens):
            super().__init__()
            self.num_tokens = num_tokens
            if in_dim == out_dim:
                self.proj = nn.Linear(in_dim, out_dim, bias=True)
            else:
                # Dimensions don't match, use linear layer with identity-like initialization
                self.proj = nn.Linear(in_dim, out_dim)
                nn.init.zeros_(self.proj.bias)
                with torch.no_grad():
                    if in_dim <= out_dim:
                        # Expand: initialize as identity for first in_dim dimensions
                        self.proj.weight.data.zero_()
                        self.proj.weight.data[:in_dim, :in_dim] = torch.eye(in_dim)
                    else:
                        # Reduce: initialize as identity for first out_dim dimensions
                        self.proj.weight.data.zero_()
                        self.proj.weight.data[:, :out_dim] = torch.eye(out_dim)

        def forward(self, x):
            return self.proj(x)

        @property
        def proj_out_num(self):
            return self.num_tokens

    direct_conn = DirectConnection(vision_hidden_size, llm_hidden_size, num_tokens)
    model.get_model().mm_projector = direct_conn
    model.config.mm_projector_type = "direct"
    model.config.proj_pooling_type = "identity"
    model.config.proj_pooling_size = 1
    model.config.num_vision_tokens = num_tokens
    model_args.mm_projector_type = "direct"
    model_args.proj_pooling_type = "identity"
    model_args.proj_pooling_size = 1
    print(f"[INFO] Installed direct mm_projector with {num_tokens} tokens.")
    return direct_conn


def has_dual_vision_encoders(state_dict):
    has_mae = any("vision_tower.mae_encoder" in k for k in state_dict.keys())
    has_clip = any("vision_tower.clip_encoder" in k for k in state_dict.keys())
    return has_mae and has_clip


def apply_model_args_to_config(model, model_args):
    """Ensure model.config mirrors model_args before rebuilding modules."""
    cfg = model.config
    cfg.image_channel = model_args.image_channel
    cfg.img_size = tuple(model_args.img_size)
    cfg.patch_size = tuple(model_args.patch_size)
    cfg.dual_vision = getattr(model_args, "dual_vision", False)
    cfg.vision_tower = model_args.vision_tower
    cfg.vision_select_layer = model_args.vision_select_layer
    cfg.vision_select_feature = model_args.vision_select_feature
    cfg.mm_projector_type = model_args.mm_projector_type
    cfg.proj_layer_type = model_args.proj_layer_type
    cfg.proj_layer_num = model_args.proj_layer_num
    cfg.proj_pooling_type = model_args.proj_pooling_type
    cfg.proj_pooling_size = model_args.proj_pooling_size


def setup_dual_vision_tower(model, model_args):
    """
    Rebuild vision_tower as MAE+CLIP dual encoders that concatenate features.
    We rely on checkpoint weights to populate parameters.
    """
    mae_encoder = build_vision_tower(model.config)
    clip_encoder = build_vision_tower(model.config)

    class DualEncoderWrapper(nn.Module):
        def __init__(self, mae_encoder, clip_encoder):
            super().__init__()
            self.mae_encoder = mae_encoder
            self.clip_encoder = clip_encoder

        def forward(self, images):
            mae_features = self.mae_encoder(images)
            clip_features = self.clip_encoder(images)
            return torch.cat([mae_features, clip_features], dim=-1)

        def __call__(self, images):
            return self.forward(images)

        @property
        def hidden_size(self):
            return self.mae_encoder.hidden_size + self.clip_encoder.hidden_size

        @property
        def dtype(self):
            return self.mae_encoder.dtype

        @property
        def device(self):
            return self.mae_encoder.device

    dual_encoder = DualEncoderWrapper(mae_encoder, clip_encoder)
    model.get_model().vision_tower = dual_encoder
    model.config.mm_hidden_size = dual_encoder.hidden_size
    model.get_model().config.mm_hidden_size = dual_encoder.hidden_size
    model.config.dual_vision = True
    model.get_model().config.dual_vision = True
    model.config.vision_tower = "vit3d_dual"
    model.get_model().config.vision_tower = "vit3d_dual"
    print(f"[INFO] Detected dual vision encoders. Combined hidden size: {dual_encoder.hidden_size}")
    return dual_encoder


def validate_mm_shapes(model, model_args):
    """
    Sanity-check that the merged dual-encoder vision output dim matches the mm_projector input,
    and that the projected token count matches the patch grid. Raises early if inconsistent.
    """
    base_model = model.get_model()
    vision = getattr(base_model, "vision_tower", None)
    mm_proj = getattr(base_model, "mm_projector", None)
    if vision is None or mm_proj is None:
        return

    vision_dim = getattr(vision, "hidden_size", None)

    def find_linear(p):
        for name in ("proj", "linear"):
            layer = getattr(p, name, None)
            if isinstance(layer, nn.Linear):
                return layer
        for layer in p.modules():
            if isinstance(layer, nn.Linear):
                return layer
        return None

    lin = find_linear(mm_proj)
    if lin is not None and vision_dim is not None and lin.in_features != vision_dim:
        raise ValueError(
            f"[ERROR] Merged checkpoint mismatch: vision hidden={vision_dim}, "
            f"mm_projector in_features={lin.in_features}. "
            "Re-export with correct dual-encoder weights so dims match."
        )

    # token-count check
    img_size = tuple(model_args.img_size)
    patch_size = tuple(model_args.patch_size)
    expected_tokens = int(np.prod([max(1, i // p) for i, p in zip(img_size, patch_size)]))
    proj_out_num = getattr(mm_proj, "proj_out_num", None)
    if proj_out_num is None:
        proj_out_num = getattr(mm_proj, "num_tokens", None)
    if proj_out_num is not None and proj_out_num != expected_tokens:
        raise ValueError(
            f"[ERROR] Merged checkpoint vision token count mismatch: "
            f"expected {expected_tokens}, got {proj_out_num}. "
            "Confirm img_size/patch_size or projector weights align with the finetuned config."
        )



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

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    model_args.vocab_size = len(tokenizer)
    print("seg_token_id: ", model_args.seg_token_id)
    print("vocab_size: ", model_args.vocab_size)

    print("Model preparation")
    if 'llama' in model_args.model_type:
        model = LamedLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )
    elif 'phi3' in model_args.model_type:
        model = LamedPhi3ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")

    model.config.seg_token_id = model_args.seg_token_id
    apply_model_args_to_config(model, model_args)

    print("Load weights with LoRA")
    if checkpoint_is_dir:
        candidates = ["model_with_lora.bin", "pytorch_model.bin", "adapter_model.bin"]
        lora_path = None
        for cand in candidates:
            path = os.path.join(checkpoint_dir, cand)
            if os.path.isfile(path):
                lora_path = path
                break
        if lora_path is None:
            raise FileNotFoundError(
                f"Directory {checkpoint_dir} does not contain any of {candidates}")
    else:
        lora_path = lora_spec_path
        if not os.path.isfile(lora_path):
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

    lora_dir = checkpoint_dir
    raw_state_dict = torch.load(lora_path, map_location="cpu")

    # (Re)build vision / projector modules to mirror dual-encoder finetuning architecture
    if has_dual_vision_encoders(raw_state_dict):
        model_args.dual_vision = True
        model_args.vision_tower = "vit3d_dual"
        setup_dual_vision_tower(model, model_args)
    else:
        raise ValueError(
            "Dual-encoder checkpoint not detected (no mae_encoder + clip_encoder keys). "
            "Use the single-encoder merge script for non-dual checkpoints."
        )

    if model_args.segmentation_module is not None:
        model.get_model().initialize_seg_modules(model_args=model_args)

    use_direct_projector = has_direct_projector_weights(raw_state_dict)
    if use_direct_projector:
        install_direct_mm_projector(model, model_args)
        state_dict = raw_state_dict
    else:
        state_dict = {}
        for k, v in raw_state_dict.items():
            new_k = k.replace("mm_projector.proj", "mm_projector.projector")
            # handle single-layer projector that saved without .0
            if new_k.endswith(".mm_projector.projector.weight"):
                new_k = new_k.replace(".projector.weight", ".projector.0.weight")
            if new_k.endswith(".mm_projector.projector.bias"):
                new_k = new_k.replace(".projector.bias", ".projector.0.bias")
            state_dict[new_k] = v

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

    if not training_args.output_dir:
        run_root = os.path.dirname(os.path.normpath(lora_dir))  # .../LaMed/output/...-no-compress
        checkpoint_name = os.path.basename(os.path.normpath(lora_dir))
        training_args.output_dir = os.path.join(run_root, "hf_merged", checkpoint_name)
        print(f"output directory: {training_args.output_dir}")

    base_state_dict = model.state_dict()
    for k, v in state_dict.items():
        base_state_dict[k] = v
    model.load_state_dict(base_state_dict, strict=True)
    validate_mm_shapes(model, model_args)

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
