import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import PreTrainedModel, PretrainedConfig, BertModel
from LaMed.src.model.multimodal_encoder.vit import ViT
from LaMed.src.utils.dist_utils import gather_features
from transformers import BertTokenizer
from LaMed.src.dataset.totalseg_labels import TOTALSEG_ID2TEXT

# Grouped organ names matching the merged masks (0-34)
# This must match GROUP_ID_TO_NAME in multi_dataset.py
GROUPED_ORGAN_ID2TEXT = {
    0: "background",
    1: "lung",
    2: "heart",
    3: "esophagus",
    4: "aorta",
    5: "brain",
    6: "adrenal_gland",
    7: "kidney",
    8: "stomach",
    9: "liver",
    10: "gallbladder",
    11: "pancreas",
    12: "spleen",
    13: "colon",
    14: "small_bowel",
    15: "urinary_bladder",
    16: "trachea",
    17: "inferior_vena_cava",
    18: "portal_vein_and_splenic_vein",
    19: "pulmonary_vein",
    20: "iliac_artery",
    21: "iliac_vena",
    22: "lumbar_vertebrae",
    23: "thoracic_vertebrae",
    24: "cervical_vertebrae",
    25: "rib",
    26: "humerus",
    27: "scapula",
    28: "clavicula",
    29: "femur",
    30: "hip",
    31: "sacrum",
    32: "gluteus",
    33: "iliopsoas",
    34: "autochthon",
}


class fVLMConfig(PretrainedConfig):
    model_type = "m3d_mae_clip_seg"

    def __init__(
        self,
        language_model_name_or_path: str = "",
        local_loss: bool = False,
        gather_loss: bool = True,
        encoder_in_channels: int = 1,
        img_size: tuple = (32, 256, 256),
        patch_size: tuple = (4, 16, 16),
        encoder_hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        spatial_dims: int = 3,
        classification: bool = True,
        text_max_length: int = 16,
        num_mask_organs: int = 1,
        clip_loss_weight: float = 1.0,
        caption_clip_loss_weight: float = 0.0,
        **kwargs,
    ):
        self.language_model_name_or_path = language_model_name_or_path
        self.encoder_in_channels = encoder_in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder_hidden_size = encoder_hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pos_embed = pos_embed
        self.spatial_dims = spatial_dims
        self.classification = classification
        self.local_loss = local_loss
        self.gather_loss = gather_loss
        self.text_max_length = text_max_length
        self.num_mask_organs = num_mask_organs
        self.clip_loss_weight = clip_loss_weight
        self.caption_clip_loss_weight = caption_clip_loss_weight
        super().__init__(**kwargs)

class fVLM(PreTrainedModel):
    config_class = fVLMConfig

    def __init__(self, config):
        super().__init__(config)
        
        self.vision_encoder = ViT(
            in_channels=config.encoder_in_channels,
            img_size=config.img_size,
            patch_size=config.patch_size,
            hidden_size=config.encoder_hidden_size,
            mlp_dim=config.encoder_hidden_size * 4,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            pos_embed=config.pos_embed,
            dropout_rate=config.dropout_rate,
            spatial_dims=config.spatial_dims,
            classification=config.classification,
            mae=False,
        )
        self.num_mask_organs = config.num_mask_organs
        
        self.classification = config.classification
        self.encoder_in_channels = config.encoder_in_channels
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        
        out_chans = self.encoder_in_channels * np.prod(self.patch_size)
        self.out_chans = out_chans
        self.encoder_hidden_size = config.encoder_hidden_size

        ### text encoder + projection ###
        self.language_encoder = BertModel.from_pretrained(config.language_model_name_or_path)
        for p in self.language_encoder.parameters():
            p.requires_grad = False
            
        self.mm_language_proj = nn.Linear(config.encoder_hidden_size, config.encoder_hidden_size)
        self.mm_vision_proj = nn.Linear(config.encoder_hidden_size, config.encoder_hidden_size)

        # logit scale (temperature) for CLIP
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1 / 0.07))  # init ~ 1/0.07 like CLIP
        )

        self.clip_loss_weight = config.clip_loss_weight
        self.caption_clip_loss_weight = config.caption_clip_loss_weight
        
        # For caption CLIP loss (like M3DCLIP)
        self.local_loss = config.local_loss
        self.gather_loss = config.gather_loss

        ## pre-tokenize all grouped organ labels (0-34) ###
        # These match the grouped masks from merged_train_masks
        text_max_length = config.text_max_length
        tokenizer = BertTokenizer.from_pretrained(config.language_model_name_or_path)
        max_grouped_id = max(GROUPED_ORGAN_ID2TEXT.keys())  # 34
        self.num_grouped_organs = max_grouped_id + 1

        # Pre-tokenize grouped organ names for fallback
        grouped_id2_input_ids = torch.zeros(
            self.num_grouped_organs,
            text_max_length,
            dtype=torch.long,
        )
        grouped_id2_attention_mask = torch.zeros_like(grouped_id2_input_ids)

        for organ_id, text in GROUPED_ORGAN_ID2TEXT.items():
            enc = tokenizer(
                text,
                max_length=text_max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            grouped_id2_input_ids[organ_id] = enc["input_ids"][0]
            grouped_id2_attention_mask[organ_id] = enc["attention_mask"][0]

        # Store as buffers so they move with the model (CPU -> GPU)
        self.register_buffer("grouped_id2_input_ids", grouped_id2_input_ids, persistent=False)
        self.register_buffer("grouped_id2_attention_mask", grouped_id2_attention_mask, persistent=False)
        
        # Also keep TotalSegmentator labels for backward compatibility
        max_id = max(TOTALSEG_ID2TEXT.keys())  # 117
        self.num_organs = max_id + 1
        
        id2_input_ids = torch.zeros(
            self.num_organs,
            text_max_length,
            dtype=torch.long,
        )
        id2_attention_mask = torch.zeros_like(id2_input_ids)

        for organ_id, text in TOTALSEG_ID2TEXT.items():
            enc = tokenizer(
                text,
                max_length=text_max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            id2_input_ids[organ_id] = enc["input_ids"][0]
            id2_attention_mask[organ_id] = enc["attention_mask"][0]

        # Store as buffers for backward compatibility
        self.register_buffer("id2_input_ids", id2_input_ids, persistent=False)
        self.register_buffer("id2_attention_mask", id2_attention_mask, persistent=False)

        # organ-aware pooling modules used for latent token selection
        # Use grouped organs for the new pipeline
        self.organ_queries = nn.Parameter(
            torch.randn(self.num_grouped_organs, 1, self.encoder_hidden_size)
        )
        nn.init.normal_(self.organ_queries, std=0.02)
        self.organ_attn = nn.MultiheadAttention(
            embed_dim=self.encoder_hidden_size,
            num_heads=config.num_heads,
            batch_first=True,
        )
        self.organ_attn_norm = nn.LayerNorm(self.encoder_hidden_size)
        organ_ffn_hidden = self.encoder_hidden_size * 4
        self.organ_ffn = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, organ_ffn_hidden),
            nn.GELU(),
            nn.Linear(organ_ffn_hidden, self.encoder_hidden_size),
        )
        self.organ_ffn_norm = nn.LayerNorm(self.encoder_hidden_size)

    def encode_text(self, input_id, attention_mask):
        text_feats = self.language_encoder(input_id, attention_mask=attention_mask)
        hidden = text_feats.last_hidden_state           # [B, L, H_txt]
        cls_feats = hidden[:, 0, :]                  # [B, H_txt]
        
        text_feats = self.mm_language_proj(cls_feats)  # [B, D_clip]
        text_feats = F.normalize(text_feats, dim=-1)   # [B, D_clip]

        return text_feats
    
    def encode_caption_text(self, input_ids, attention_mask):
        """Encode caption text for CLIP loss (similar to M3DCLIP.encode_text)"""
        text_feats = self.language_encoder(input_ids, attention_mask=attention_mask)["last_hidden_state"]
        text_feats = self.mm_language_proj(text_feats)  # [B, L, D_clip]
        text_feats = F.normalize(text_feats, dim=-1)  # [B, L, D_clip]
        # Use CLS token (first token) for sentence-level CLIP, matching M3DCLIP.forward line 272
        text_feats = text_feats[:, 0, :]  # [B, D_clip]
        return text_feats

    def _build_organ_mask_indices(self, seg):
        B, _, D, H, W = seg.shape
        device = seg.device

        seg_ = seg.long()

        # masks for MAE (union of selected organs)
        mae_masks = torch.zeros(
            (B, 1, D, H, W), dtype=torch.float32, device=device
        )

        clip_mask_entries = []
        clip_batch_indices = []
        clip_organ_ids = []
        
        for b in range(B):
            seg_b_vox = seg_[b, 0]  # [D,H,W]
            
            organ_labels = torch.unique(seg_b_vox)
            organ_labels = organ_labels[organ_labels > 0]
            
            if organ_labels.numel() == 0:
                continue

            if self.num_mask_organs > 0:
                K = min(self.num_mask_organs, organ_labels.numel())
                perm = torch.randperm(organ_labels.numel(), device=device)
                selected_ids = organ_labels[perm[:K]]
            else:
                selected_ids = organ_labels
            
            mae_mask_b = torch.zeros_like(seg_b_vox, dtype=torch.bool)
            for oid in selected_ids:
                organ_mask = (seg_b_vox == oid)
                mae_mask_b |= organ_mask
                clip_mask_entries.append(organ_mask.unsqueeze(0).float())
                clip_batch_indices.append(b)
                clip_organ_ids.append(oid)
            mae_masks[b, 0] = mae_mask_b

        if clip_mask_entries:
            clip_token_masks = torch.stack(clip_mask_entries, dim=0)
            clip_batch_indices = torch.tensor(clip_batch_indices, device=device, dtype=torch.long)
            clip_organ_ids = torch.stack(clip_organ_ids).to(device=device, dtype=torch.long)
        else:
            clip_token_masks = torch.zeros((0, 1, D, H, W), device=device, dtype=torch.float32)
            clip_batch_indices = torch.zeros(0, device=device, dtype=torch.long)
            clip_organ_ids = torch.zeros(0, device=device, dtype=torch.long)

        return mae_masks, clip_token_masks, clip_batch_indices, clip_organ_ids
    
    def _select_organ_text(
        self,
        clip_batch_indices,
        clip_organ_ids,
        organ_input_ids=None,
        organ_attention_mask=None,
        organ_batch_indices=None,
        organ_ids=None,
        organ_normal_flags=None,
    ):
        """
        Match organ-specific text tokens to the organ masks selected for CLIP.
        Falls back to pre-tokenized grouped organ labels if no organ text is provided.
        """
        # If no organ-specific text provided, use grouped organ labels as fallback
        if (
            organ_input_ids is None
            or organ_attention_mask is None
            or organ_batch_indices is None
            or organ_ids is None
            or organ_input_ids.numel() == 0
        ):
            # Use grouped organ IDs (0-34) for indexing
            text_input_ids = self.grouped_id2_input_ids[clip_organ_ids]
            text_attn_mask = self.grouped_id2_attention_mask[clip_organ_ids]
            keep_indices = torch.arange(
                clip_organ_ids.size(0), device=clip_organ_ids.device, dtype=torch.long
            )
            normal_flags = torch.zeros_like(keep_indices, dtype=torch.bool)
            return text_input_ids, text_attn_mask, keep_indices, normal_flags

        # Match provided organ text to visual features
        matched_vis_indices = []
        matched_text_inputs = []
        matched_text_masks = []
        matched_normal_flags = []

        for idx, (b_idx, o_id) in enumerate(zip(clip_batch_indices.tolist(), clip_organ_ids.tolist())):
            match = ((organ_batch_indices == b_idx) & (organ_ids == o_id)).nonzero(as_tuple=False)
            if match.numel() == 0:
                # Fallback to grouped organ label for this specific organ
                matched_vis_indices.append(idx)
                matched_text_inputs.append(self.grouped_id2_input_ids[o_id])
                matched_text_masks.append(self.grouped_id2_attention_mask[o_id])
                matched_normal_flags.append(False)
                continue
            text_idx = match[0, 0].item()
            matched_vis_indices.append(idx)
            matched_text_inputs.append(organ_input_ids[text_idx])
            matched_text_masks.append(organ_attention_mask[text_idx])
            if organ_normal_flags is not None and organ_normal_flags.numel() > text_idx:
                matched_normal_flags.append(bool(organ_normal_flags[text_idx].item()))
            else:
                matched_normal_flags.append(False)

        if not matched_vis_indices:
            return None, None, None, None

        text_input_ids = torch.stack(matched_text_inputs, dim=0).to(clip_batch_indices.device)
        text_attn_mask = torch.stack(matched_text_masks, dim=0).to(clip_batch_indices.device)
        keep_indices = torch.tensor(matched_vis_indices, device=clip_batch_indices.device, dtype=torch.long)
        normal_flags = torch.tensor(matched_normal_flags, device=clip_batch_indices.device, dtype=torch.bool)
        return text_input_ids, text_attn_mask, keep_indices, normal_flags

    def forward(self, image, seg, organ_input_ids=None, organ_attention_mask=None, organ_batch_indices=None, organ_ids=None, organ_normal_flags=None, input_ids=None, attention_mask=None, labels=None, return_image=False):
        batch_size = image.size(0)
        in_chanl = image.size(1)
        assert in_chanl == self.encoder_in_channels

        # keep original image for targets
        orig_image = image.clone()
        masked_image = None

        # 1) build per-organ masks/metadata for CLIP (no MAE reconstruction)
        mae_masks, clip_token_masks, clip_batch_indices, clip_organ_ids = self._build_organ_mask_indices(seg)
        clip_token_masks = clip_token_masks.to(image.dtype)

        ### ------------------CLIP loss 1: organ-name contrastive (TotalSegmentator)------------------ ###
        # Anchor the loss to a trainable parameter so it always has grad_fn
        clip_loss_organ = self.mm_vision_proj.weight.sum() * 0.0

        num_clip_entries = clip_token_masks.size(0)
        # If a rank sees no valid organs, none of the model params would be used
        # when ddp_find_unused_parameters=False. Touch all trainable params with
        # a zero-scale dummy term so every bucket participates in allreduce.
        if num_clip_entries == 0:
            dummy = torch.zeros((), device=image.device, dtype=image.dtype)
            for p in self.parameters():
                if p.requires_grad:
                    dummy = dummy + p.sum() * 0.0
            total_loss = self.clip_loss_weight * (clip_loss_organ + dummy)
            return {
                "loss": total_loss,
                "clip_loss_organ": clip_loss_organ,
            }
        if self.clip_loss_weight > 0 and num_clip_entries > 0:
            # Prefer organ-specific text from the dataset; fall back to TotalSegmentator labels.
            text_input_ids, text_attn_mask, keep_indices, normal_flags = self._select_organ_text(
                clip_batch_indices=clip_batch_indices,
                clip_organ_ids=clip_organ_ids,
                organ_input_ids=organ_input_ids,
                organ_attention_mask=organ_attention_mask,
                organ_batch_indices=organ_batch_indices,
                organ_ids=organ_ids,
                organ_normal_flags=organ_normal_flags,
            )
            if keep_indices is None:
                text_input_ids = self.id2_input_ids[clip_organ_ids]        # [N_clip, L]
                text_attn_mask = self.id2_attention_mask[clip_organ_ids]   # [N_clip, L]
                keep_indices = torch.arange(
                    num_clip_entries, device=image.device, dtype=torch.long
                )
                normal_flags = torch.zeros_like(keep_indices, dtype=torch.bool)

            if keep_indices.numel() == 0:
                clip_loss_organ = clip_loss_organ + 0.0
            else:
                # Align visual tokens with the matched organ text entries.
                clip_token_masks = clip_token_masks.index_select(0, keep_indices)
                clip_batch_indices = clip_batch_indices.index_select(0, keep_indices)
                clip_organ_ids = clip_organ_ids.index_select(0, keep_indices)
                normal_flags = normal_flags.index_select(0, keep_indices)
                text_input_ids = text_input_ids.to(image.device)
                text_attn_mask = text_attn_mask.to(image.device)
                num_clip_entries = clip_token_masks.size(0)
                text_emb = self.encode_text(text_input_ids, text_attn_mask)  # [B_valid, D]

            # encode the full volume once and keep only spatial tokens
                enc_all_full, _ = self.vision_encoder(image)        # [B, N, H_enc']
                if self.classification:
                    enc_all_full = enc_all_full[:, 1:, :]

                # convert the voxel-level organ mask into patch-space indicators (match fvlm-main: max-pool)
                organ_token_mask = F.max_pool3d(
                    clip_token_masks,
                    kernel_size=tuple(self.patch_size),
                    stride=tuple(self.patch_size),
                )
                organ_token_mask = organ_token_mask.squeeze(1) > 0.0      # [N_clip, D',H',W']
                organ_token_mask = organ_token_mask.view(num_clip_entries, -1)  # [N_clip, N]

                vis_tokens_all = enc_all_full[clip_batch_indices]                 # [N_clip, N, H_enc']
                organ_query_tokens = self.organ_queries.index_select(0, clip_organ_ids)  # [N_clip, 1, H_enc']

                pooled_embeds = []
                for idx in range(num_clip_entries):
                    token_mask = organ_token_mask[idx]
                    organ_tokens = vis_tokens_all[idx][token_mask]  # [N_tokens, H_enc']
                    if organ_tokens.shape[0] == 0:
                        # fallback: average full grid if mask collapsed (rare but safe)
                        organ_tokens = vis_tokens_all[idx].mean(dim=0, keepdim=True)

                    # match fvlm-main style: query attends over ROI tokens (no self-attn+FFN stack)
                    query_token = organ_query_tokens[idx].unsqueeze(0)  # [1,1,H_enc']
                    key_val = organ_tokens.unsqueeze(0)                # [1,N_tokens,H_enc']
                    attn_out, _ = self.organ_attn(query_token, key_val, key_val)  # [1,1,H_enc']
                    pooled_embeds.append(attn_out.squeeze(0).squeeze(0))  # [H_enc']

                vis_emb = torch.stack(pooled_embeds, dim=0)          # [N_clip, H_enc']
                vis_emb = self.mm_vision_proj(vis_emb)               # [B_valid, D]
                vis_emb = F.normalize(vis_emb, dim=-1)               # [B_valid, D]

                B_valid = vis_emb.size(0)
                logit_scale = self.logit_scale.exp()

                # To mirror fvlm-main: compute organ-wise ITC and sum.
                clip_loss_list = []
                for oid in torch.unique(clip_organ_ids):
                    organ_mask = clip_organ_ids == oid
                    if organ_mask.sum() == 0:
                        continue

                    o_vis = vis_emb[organ_mask]
                    o_text = text_emb[organ_mask]
                    o_text_ids = text_input_ids[organ_mask]
                    o_nf = normal_flags[organ_mask] if normal_flags is not None and normal_flags.numel() == B_valid else None

                    logits_per_image = logit_scale * (o_vis @ o_text.t())
                    logits_per_text = logit_scale * (o_text @ o_vis.t())

                    targets = torch.zeros(
                        (o_vis.size(0), o_vis.size(0)),
                        device=image.device,
                        dtype=logits_per_image.dtype,
                    )
                    targets.fill_diagonal_(1.0)

                    if o_nf is not None and o_nf.numel() == o_vis.size(0):
                        nf = o_nf.float()
                        normal_matrix = (nf[:, None] * nf[None, :]).bool()
                        normal_matrix.fill_diagonal_(False)
                        targets = targets + normal_matrix.float()

                    targets = targets / torch.clamp(targets.sum(dim=1, keepdim=True), min=1e-6)

                    log_probs_i2t = F.log_softmax(logits_per_image, dim=1)
                    log_probs_t2i = F.log_softmax(logits_per_text, dim=1)

                    loss_i2t = -torch.sum(targets * log_probs_i2t, dim=1).mean()
                    loss_t2i = -torch.sum(targets * log_probs_t2i, dim=1).mean()
                    clip_loss_list.append(0.5 * (loss_i2t + loss_t2i))

                if clip_loss_list:
                    clip_loss_organ = sum(clip_loss_list) / len(clip_loss_list)

        # Only organ-level CLIP contributes to the loss.
        total_loss = self.clip_loss_weight * clip_loss_organ

        return {
            "loss": total_loss,
            "clip_loss_organ": clip_loss_organ,
        }