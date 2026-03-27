from abc import ABC, abstractmethod
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_mm_projector
from .segmentation_module.builder import build_segmentation_module
from LaMed.src.model.loss import BCELoss, BinaryDiceLoss

from safetensors.torch import load_file


def _prepare_vision_state_dict(raw_state):
    """
    Normalize checkpoints so that only the vision encoder weights remain and
    their keys match the local ViT implementation.
    """
    if isinstance(raw_state, dict) and "model" in raw_state and isinstance(raw_state["model"], dict):
        raw_state = raw_state["model"]

    prefixes = ("vision_encoder.", "visual_encoder.")
    filtered = {}
    for key, value in raw_state.items():
        if not isinstance(value, torch.Tensor):
            continue
        matched = False
        for prefix in prefixes:
            if key.startswith(prefix):
                filtered_key = key[len(prefix):]
                filtered[filtered_key] = value
                matched = True
                break
        if matched:
            continue
        # Some checkpoints may already be stripped; keep those as-is.
        if any(key.startswith(p.replace(".", "")) for p in prefixes):
            filtered[key] = value

    return filtered or raw_state


def _enumerate_grid_shapes(num_tokens):
    combos = []
    max_dim = int(round(num_tokens ** (1 / 3))) + 4
    for d in range(1, max_dim):
        if num_tokens % d != 0:
            continue
        remainder = num_tokens // d
        max_h = int(round(math.sqrt(remainder))) + 4
        for h in range(1, max_h):
            if remainder % h != 0:
                continue
            w = remainder // h
            combos.append((d, h, w))
    return combos


def _pick_grid_shape(num_tokens, patch_size, ref_img_size):
    combos = _enumerate_grid_shapes(num_tokens)
    if not combos:
        return None
    if ref_img_size is None:
        combos.sort(key=lambda dims: max(dims) - min(dims))
        return combos[0]

    ref_img = tuple(int(x) for x in ref_img_size)
    best_perm = None
    best_score = float("inf")
    for dims in combos:
        for perm in set(itertools.permutations(dims)):
            candidate_img = tuple(int(perm[i] * patch_size[i]) for i in range(3))
            score = sum((candidate_img[i] - ref_img[i]) ** 2 for i in range(3))
            if score < best_score:
                best_score = score
                best_perm = perm
    return best_perm or combos[0]


def _infer_img_size_from_state(state_dict, patch_size, ref_img_size):
    pos_key = "patch_embedding.position_embeddings"
    if pos_key not in state_dict or patch_size is None:
        return None, None
    num_tokens = state_dict[pos_key].shape[1]
    grid_shape = _pick_grid_shape(num_tokens, patch_size, ref_img_size)
    if grid_shape is None:
        return None, None
    inferred_img = tuple(int(grid_shape[i] * patch_size[i]) for i in range(3))
    return inferred_img, grid_shape


def _resize_position_embeddings(src_pe, source_grid, target_grid, dst_param):
    if src_pe.shape == dst_param.shape:
        return src_pe
    if source_grid is None or target_grid is None:
        return src_pe

    pe = src_pe.to(dtype=dst_param.dtype)
    pe = pe.transpose(1, 2).reshape(1, pe.shape[2], *source_grid)
    pe = F.interpolate(pe, size=target_grid, mode="trilinear", align_corners=False)
    pe = pe.reshape(1, pe.shape[1], -1).transpose(1, 2)
    return pe

class LamedMetaModel:
    def __init__(self, config):
        super(LamedMetaModel, self).__init__(config)

        self.config = config
        self.seg_enable = False

        if hasattr(config, "vision_tower"):
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_mm_projector(config)

        if hasattr(config, "segmentation_module") and config.segmentation_module is not None:
            self.seg_enable = True
            self.seg_module = build_segmentation_module(config)

            self.seg_projector = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_size, config.mm_hidden_size),
                nn.Dropout(0.1),
            )

            self.dice_loss = BinaryDiceLoss()
            self.bce_loss = BCELoss()

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        return vision_tower

    def initialize_vision_modules(self, model_args):
        self.config.image_channel = model_args.image_channel
        self.config.img_size = model_args.img_size
        self.config.patch_size = model_args.patch_size

        self.config.vision_tower = model_args.vision_tower
        self.config.vision_select_layer = model_args.vision_select_layer
        self.config.vision_select_feature = model_args.vision_select_feature

        # Auto-configure fvlm vision tower to match checkpoint
        is_fvlm = model_args.vision_tower and "fvlm" in model_args.vision_tower.lower()
        if is_fvlm:
            # fvlm checkpoint uses patch_size=(16, 16, 32) and img_size=(112, 256, 352)
            model_args.patch_size = (16, 16, 32)
            model_args.img_size = (112, 256, 352)
            self.config.patch_size = (16, 16, 32)
            self.config.img_size = (112, 256, 352)
            print(f"[Vision] Auto-configured fvlm: patch_size={self.config.patch_size}, img_size={self.config.img_size}")

        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.proj_layer_type = model_args.proj_layer_type
        self.config.proj_layer_num = model_args.proj_layer_num
        self.config.proj_pooling_type = model_args.proj_pooling_type
        self.config.proj_pooling_size = model_args.proj_pooling_size

        vision_ckpt = None
        ckpt_grid = None

        if model_args.pretrain_vision_model is not None:
            if model_args.pretrain_vision_model.endswith(".safetensors"):
                raw_state = load_file(model_args.pretrain_vision_model, device="cpu")
            else:
                raw_state = torch.load(model_args.pretrain_vision_model, map_location="cpu")
            vision_ckpt = _prepare_vision_state_dict(raw_state)

            patch_weight = vision_ckpt.get("patch_embedding.patch_embeddings.weight")
            if patch_weight is not None and patch_weight.ndim >= 3:
                raw_patch = patch_weight.shape[2:]
                if len(raw_patch) == 3:
                    inferred_patch = tuple(int(dim) for dim in raw_patch)
                    if inferred_patch != tuple(model_args.patch_size):
                        # For fvlm, preserve the explicitly set values; for others, use inferred
                        if not is_fvlm:
                            model_args.patch_size = inferred_patch
                            self.config.patch_size = inferred_patch
                        else:
                            print(f"[Vision] fvlm patch_size preserved as {model_args.patch_size} (checkpoint has {inferred_patch})")

            inferred_img, ckpt_grid = _infer_img_size_from_state(
                vision_ckpt,
                tuple(model_args.patch_size),
                tuple(self.config.img_size),
            )

        target_signature = (tuple(self.config.img_size), tuple(self.config.patch_size))
        current_signature = getattr(self, "_vision_cfg_signature", None)

        if self.get_vision_tower() is None or current_signature != target_signature:
            self.vision_tower = build_vision_tower(self.config)
            self._vision_cfg_signature = target_signature

        # If you have a more robust vision encoder, try freezing the vision tower by requires_grad_(False)
        self.vision_tower.requires_grad_(not model_args.freeze_vision_tower)

        if vision_ckpt is not None:
            pos_key = "patch_embedding.position_embeddings"
            target_grid = tuple(max(1, self.config.img_size[i] // self.config.patch_size[i]) for i in range(3))
            if pos_key in vision_ckpt:
                resized = _resize_position_embeddings(
                    vision_ckpt[pos_key],
                    ckpt_grid,
                    target_grid,
                    self.vision_tower.vision_tower.patch_embedding.position_embeddings,
                )
                vision_ckpt[pos_key] = resized

            missing, unexpected = self.vision_tower.vision_tower.load_state_dict(vision_ckpt, strict=False)
            print(f"Loaded pretrained vision model from {model_args.pretrain_vision_model}")
            if missing:
                print(f"[Vision] Missing keys: {missing}")
            if unexpected:
                print(f"[Vision] Unexpected keys: {unexpected}")

        self.config.mm_hidden_size = self.vision_tower.hidden_size

        # mm_projector
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_mm_projector(self.config)

        if model_args.pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=True)

    def initialize_seg_modules(self, model_args):
        self.config.segmentation_module = model_args.segmentation_module

        # segmentation_module
        if getattr(self, 'seg_module', None) is None:
            self.seg_module = build_segmentation_module(self.config)
            self.seg_projector = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.hidden_size, self.config.mm_hidden_size),
                nn.Dropout(0.1),
            )
            self.seg_enable = True

        if model_args.pretrain_seg_module is not None:
            seg_module_weights = torch.load(model_args.pretrain_seg_module, map_location='cpu')
            new_state_dict = {}
            for key, value in seg_module_weights.items():
                if key.startswith('model.text_encoder.') or key.startswith('text_encoder.'):
                    continue
                if key.startswith('model.'):
                    new_key = key[len('model.'):]
                    new_state_dict[new_key] = value
            self.seg_module.load_state_dict(new_state_dict, strict=True)

        self.dice_loss = BinaryDiceLoss()
        self.bce_loss = BCELoss()

class LamedMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        else:
            image_features = self.encode_images(images)
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            inputs_embeds = torch.cat(
                (inputs_embeds[:, :1, :], image_features, inputs_embeds[:, (image_features.shape[1] + 1):, :]), dim=1)
            
            # Expand attention_mask to include vision tokens
            if attention_mask is not None:
                batch_size = attention_mask.shape[0]
                num_vision_tokens = image_features.shape[1]
                # Create attention mask: [1 (BOS), num_vision_tokens (all 1s), original text mask after placeholders]
                vision_attention = torch.ones(
                    (batch_size, num_vision_tokens),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                # BOS token attention (always 1)
                bos_attention = torch.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                # Skip the placeholder tokens that were replaced by vision features
                if attention_mask.shape[1] > (num_vision_tokens + 1):
                    text_attention = attention_mask[:, (num_vision_tokens + 1):]
                else:
                    text_attention = attention_mask[:, 0:0]
                # Concatenate: [BOS, vision_tokens, text_tokens]
                attention_mask = torch.cat(
                    (bos_attention, vision_attention, text_attention), dim=1
                )
            
            # Expand labels to match the new sequence length (including vision tokens)
            if labels is not None:
                batch_size = labels.shape[0]
                num_vision_tokens = image_features.shape[1]
                # Match the structure of inputs_embeds exactly:
                # inputs_embeds = [BOS, vision_tokens, text_tokens]
                # where text_tokens = inputs_embeds[:, (num_vision_tokens + 1):, :]
                
                # Get BOS label (matches inputs_embeds[:, :1, :])
                bos_label = labels[:, :1]  # [batch, 1]
                
                # Get text labels (matches inputs_embeds[:, (num_vision_tokens + 1):, :])
                # This skips BOS (position 0) and <im_patch> positions (1 to num_vision_tokens)
                if labels.shape[1] > (num_vision_tokens + 1):
                    # Normal case: we have enough positions
                    text_labels = labels[:, (num_vision_tokens + 1):]
                else:
                    # Original labels are truncated (shorter than num_vision_tokens + 1)
                    # This happens when max_length is too small
                    # Fallback: try to extract answer labels from the end of the original labels
                    # The answer labels (non--100) should be at the end
                    # Find the last non--100 position
                    non_ignore_mask = (labels != -100)
                    if non_ignore_mask.any():
                        # Get the last part that has non--100 values (likely answer labels)
                        last_non_ignore = non_ignore_mask.int().argmax(dim=1).max().item()
                        if last_non_ignore > 0:
                            # Take from the last non--100 position onwards
                            text_labels = labels[:, max(1, last_non_ignore):]
                        else:
                            # All labels are -100, use empty
                            text_labels = labels[:, labels.shape[1]:]
                    else:
                        # All labels are -100, use what we have after BOS
                        text_labels = labels[:, 1:] if labels.shape[1] > 1 else labels[:, labels.shape[1]:]
                
                # Create vision token labels (all -100, ignore index)
                vision_labels = torch.full(
                    (batch_size, num_vision_tokens),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                
                # Concatenate: [BOS, vision_tokens, text_tokens]
                labels = torch.cat(
                    (bos_label, vision_labels, text_labels), dim=1
                )
        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            else:
                # we add 4 new tokens
                # if new tokens need input, please train input_embeddings
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                # if new tokens need predict, please train output_embeddings
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']

            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings = embed_tokens_weight
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")