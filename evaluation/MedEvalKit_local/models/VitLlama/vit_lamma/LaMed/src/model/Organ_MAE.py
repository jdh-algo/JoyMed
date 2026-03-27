import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from transformers import PreTrainedModel, PretrainedConfig, BertModel, BertTokenizer

from LaMed.src.model.multimodal_encoder.vit import ViT
from LaMed.src.model.multimodal_encoder.mae_vit import MAEViTDecoder
from LaMed.src.dataset.totalseg_labels import TOTALSEG_ID2NAME_CLINICAL
from LaMed.src.model.CLIP import (
    build_3d_sincos_position_embedding,
    patchify_image,
    unpatchify,
    batched_shuffle_indices,
)

class OrganMAEConfig(PretrainedConfig):
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
        decoder_hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        mask_ratio: float = 0.75,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        classification: bool = True,
        text_max_length: int = 16,
        num_mask_organs: int = 1,
        num_clip_organs: int = 0,
        clip_loss_weight: float = 0.5,
        caption_clip_loss_weight: float = 0.5,
        mae_loss_weight: float = 1.0,
        regular_mae_loss_weight: float = 0.0,
        **kwargs,
    ):
        self.language_model_name_or_path = language_model_name_or_path
        self.encoder_in_channels = encoder_in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pos_embed = pos_embed
        self.dropout_rate = dropout_rate
        self.spatial_dims = spatial_dims
        self.mask_ratio = mask_ratio
        self.classification=classification
        self.text_max_length=text_max_length
        self.num_mask_organs=num_mask_organs
        self.num_clip_organs=num_clip_organs
        self.clip_loss_weight=clip_loss_weight
        self.caption_clip_loss_weight = caption_clip_loss_weight
        self.mae_loss_weight = mae_loss_weight
        self.regular_mae_loss_weight = regular_mae_loss_weight
        self.local_loss = local_loss
        self.gather_loss = gather_loss
        super().__init__(**kwargs)

class OrganMAE(PreTrainedModel):
    config_class = OrganMAEConfig

    def __init__(self, config):
        super().__init__(config)
        
        self.vision_encoder = ViT(
            in_channels=config.encoder_in_channels,
            img_size=config.img_size,
            patch_size=config.patch_size,
            hidden_size=config.encoder_hidden_size,
            mlp_dim=config.mlp_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            pos_embed=config.pos_embed,
            dropout_rate=config.dropout_rate,
            spatial_dims=config.spatial_dims,
            classification=config.classification,
            mae=True
        )
        self.num_mask_organs = config.num_mask_organs
        self.num_clip_organs = config.num_clip_organs
        
        self.classification = config.classification
        self.mask_ratio = config.mask_ratio
        self.encoder_in_channels = config.encoder_in_channels
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        
        out_chans = self.encoder_in_channels * np.prod(self.patch_size)
        self.out_chans = out_chans

        self.encoder_hidden_size = config.encoder_hidden_size
        self.decoder_hidden_size = config.decoder_hidden_size
        
        self.vision_decoder = MAEViTDecoder(patch_size=self.patch_size,
                               num_classes=self.out_chans,
                               embed_dim=self.decoder_hidden_size,
                               depth=config.num_layers,
                               num_heads=config.num_heads)

        self.encoder_to_decoder = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_hidden_size))
        self.patch_norm = nn.LayerNorm(normalized_shape=(self.out_chans,), eps=1e-6, elementwise_affine=False)
        self.criterion = nn.MSELoss(reduction='sum')
        
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        nn.init.normal_(self.mask_token, std=0.02)

        grid_size = []
        for in_size, pa_size in zip(self.img_size, self.patch_size):
            assert in_size % pa_size == 0, "input size and patch size are not proper"
            grid_size.append(in_size // pa_size)
        self.grid_size = grid_size

        # build positional encoding for decoder
        if config.pos_embed == 'sincos':
            with torch.no_grad():
                self.decoder_pos_embed = build_3d_sincos_position_embedding(grid_size, 
                                                                            self.decoder_hidden_size, 
                                                                            num_tokens=0)
        elif config.pos_embed == 'perceptron':
            with torch.no_grad():
                self.decoder_pos_embed = build_3d_sincos_position_embedding(grid_size, 
                                                                            self.decoder_hidden_size, 
                                                                            num_tokens=0)

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

        # weight for CLIP loss relative to MAE loss
        self.clip_loss_weight = config.clip_loss_weight
        self.caption_clip_loss_weight = config.caption_clip_loss_weight
        self.mae_loss_weight = config.mae_loss_weight
        self.regular_mae_loss_weight = config.regular_mae_loss_weight
        
        # For caption CLIP loss (like M3DCLIP)
        self.local_loss = config.local_loss
        self.gather_loss = config.gather_loss

        # Separate criterion for regular MAE to mirror baseline implementation
        self.regular_mae_criterion = nn.MSELoss()

        ## pre-tokenize all 117 TotalSegmentator labels ###
        # do this once in __init__
        text_max_length = config.text_max_length
        tokenizer = BertTokenizer.from_pretrained(config.language_model_name_or_path)
        max_id = max(TOTALSEG_ID2NAME_CLINICAL.keys())  # 117
        self.num_organs = max_id + 1

        id2_input_ids = torch.zeros(
            self.num_organs,
            text_max_length,
            dtype=torch.long,
        )
        id2_attention_mask = torch.zeros_like(id2_input_ids)

        for organ_id, text in TOTALSEG_ID2NAME_CLINICAL.items():
            enc = tokenizer(
                text,
                max_length=text_max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            id2_input_ids[organ_id] = enc["input_ids"][0]
            id2_attention_mask[organ_id] = enc["attention_mask"][0]

        # store as buffers so they move with the model (CPU -> GPU)
        self.register_buffer("id2_input_ids", id2_input_ids, persistent=False)
        self.register_buffer("id2_attention_mask", id2_attention_mask, persistent=False)

        # organ-aware pooling modules used for latent token selection
        self.organ_queries = nn.Parameter(
            torch.randn(self.num_organs, 1, self.encoder_hidden_size)
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

            # --- MAE organ selection (unchanged behavior) ---
            if self.num_mask_organs > 0:
                K = min(self.num_mask_organs, organ_labels.numel())
                perm = torch.randperm(organ_labels.numel(), device=device)
                mae_selected_ids = organ_labels[perm[:K]]
            else:
                mae_selected_ids = organ_labels
            
            # --- CLIP organ selection (decoupled from MAE) ---
            # num_clip_organs <= 0 means use ALL available organs
            if self.num_clip_organs > 0:
                K_clip = min(self.num_clip_organs, organ_labels.numel())
                perm_clip = torch.randperm(organ_labels.numel(), device=device)
                clip_selected_ids = organ_labels[perm_clip[:K_clip]]
            else:
                # Use ALL organs for CLIP
                clip_selected_ids = organ_labels
            
            # Build MAE mask from MAE-selected organs
            mae_mask_b = torch.zeros_like(seg_b_vox, dtype=torch.bool)
            for oid in mae_selected_ids:
                organ_mask = (seg_b_vox == oid)
                mae_mask_b |= organ_mask
            mae_masks[b, 0] = mae_mask_b
            
            # Build CLIP entries from CLIP-selected organs (independently)
            for oid in clip_selected_ids:
                organ_mask = (seg_b_vox == oid)
                clip_mask_entries.append(organ_mask.unsqueeze(0).float())
                clip_batch_indices.append(b)
                clip_organ_ids.append(oid)

        if clip_mask_entries:
            clip_token_masks = torch.stack(clip_mask_entries, dim=0)
            clip_batch_indices = torch.tensor(clip_batch_indices, device=device, dtype=torch.long)
            clip_organ_ids = torch.stack(clip_organ_ids).to(device=device, dtype=torch.long)
        else:
            clip_token_masks = torch.zeros((0, 1, D, H, W), device=device, dtype=torch.float32)
            clip_batch_indices = torch.zeros(0, device=device, dtype=torch.long)
            clip_organ_ids = torch.zeros(0, device=device, dtype=torch.long)

        return mae_masks, clip_token_masks, clip_batch_indices, clip_organ_ids

    def forward(self, image, seg, input_ids=None, attention_mask=None, labels=None, return_image=False):
        batch_size = image.size(0)
        in_chanl = image.size(1)
        assert in_chanl == self.encoder_in_channels

        # keep original image for targets
        orig_image = image.clone()
        masked_image = None

        # 1) remove the selected organ from the INPUT image + get organ masks / indices
        # Build per-volume union masks for MAE plus per-organ masks/metadata for CLIP.
        mae_masks, clip_token_masks, clip_batch_indices, clip_organ_ids = self._build_organ_mask_indices(seg)
        # Keep dtypes consistent with the incoming image for downstream math.
        mae_masks = mae_masks.to(image.dtype)
        clip_token_masks = clip_token_masks.to(image.dtype)
        # organ_masks:  [B,1,D,H,W] (1 inside organ)

        mse_loss = torch.tensor(0.0, device=image.device, dtype=image.dtype)
        recon_image = image

        # Regular MAE loss (random patch masking, independent from organ masking)
        regular_mae_loss = torch.tensor(0.0, device=image.device, dtype=image.dtype)
        if self.regular_mae_loss_weight > 0:
            tokens = self.vision_encoder.patch_embedding(image)  # [B, N, H_enc]
            length = np.prod(self.grid_size)
            sel_length = int(length * (1 - self.mask_ratio))
            msk_length = length - sel_length

            shuffle_indices = batched_shuffle_indices(batch_size, length, device=image.device)
            sel_idx = shuffle_indices[:, :sel_length]
            msk_idx = shuffle_indices[:, sel_length:]

            sel_tokens = tokens.gather(
                dim=1, index=sel_idx[..., None].expand(-1, -1, tokens.size(-1))
            )

            enc_vis, _ = self.vision_encoder(sel_tokens)
            if self.classification:
                enc_vis = enc_vis[:, 1:, :]
            dec_vis = self.encoder_to_decoder(enc_vis)

            dec_mask = self.mask_token.expand(batch_size, msk_length, -1)
            dec_all = torch.cat([dec_vis, dec_mask], dim=1)

            dec_pos = self.decoder_pos_embed.expand(batch_size, -1, -1).gather(
                1, shuffle_indices[..., None].expand(-1, -1, self.decoder_hidden_size)
            )
            dec_all = dec_all + dec_pos
            pred = self.vision_decoder(dec_all)

            pix = patchify_image(image, self.patch_size)  # [B, N, out_chans]
            tgt = pix.gather(1, msk_idx[..., None].expand(-1, -1, pix.size(-1)))

            regular_mae_loss = self.regular_mae_criterion(
                pred[:, -msk_length:, :], self.patch_norm(tgt.detach())
            )

        if self.mae_loss_weight > 0:
            tokens_all = self.vision_encoder.patch_embedding(image)  # [B, N, H_enc]
            patchified = patchify_image(image, self.patch_size)      # [B, N, out_chans]
            masked_patch_tokens = patchified.clone() if return_image else None

            # Compute per-patch occupancy from the organ mask.
            patch_coverage = F.avg_pool3d(
                mae_masks,
                kernel_size=tuple(self.patch_size),
                stride=tuple(self.patch_size),
            )
            patch_coverage = patch_coverage.view(batch_size, -1)
            eligible_patch_mask = patch_coverage > 0.0

            loss_sum = image.new_tensor(0.0)
            denom = image.new_tensor(0.0)
            recon_patches = None
            if return_image:
                recon_patches = torch.zeros_like(patchified)

            total_masked_patches = 0
            total_patches = batch_size * patch_coverage.size(1)

            for b in range(batch_size):
                mask_idx = torch.nonzero(eligible_patch_mask[b], as_tuple=False).flatten()
                if mask_idx.numel() == 0:
                    if return_image:
                        recon_patches[b:b+1] = self.patch_norm(patchified[b:b+1])
                    continue

                keep_idx = torch.nonzero(~eligible_patch_mask[b], as_tuple=False).flatten()

                mask_perm = mask_idx[
                    torch.randperm(mask_idx.numel(), device=mask_idx.device)
                ]
                if keep_idx.numel() == 0:
                    # Guarantee at least one visible token for the encoder.
                    keep_idx = mask_perm[:1]
                    mask_perm = mask_perm[1:]

                if mask_perm.numel() == 0:
                    if return_image:
                        recon_patches[b:b+1] = self.patch_norm(patchified[b:b+1])
                    continue

                if keep_idx.numel() > 0:
                    sel_perm = keep_idx[
                        torch.randperm(keep_idx.numel(), device=keep_idx.device)
                    ]
                else:
                    sel_perm = keep_idx

                tokens_vis = tokens_all[b:b+1].index_select(1, sel_perm)
                enc_vis, _ = self.vision_encoder(tokens_vis)
                if self.classification:
                    enc_vis = enc_vis[:, 1:, :]

                dec_vis = self.encoder_to_decoder(enc_vis)
                dec_mask = self.mask_token.expand(1, mask_perm.numel(), -1)
                dec_all = torch.cat([dec_vis, dec_mask], dim=1)

                shuffle_idx = torch.cat([sel_perm, mask_perm], dim=0)
                dec_pos = self.decoder_pos_embed.index_select(1, shuffle_idx)
                dec_all = dec_all + dec_pos

                pred = self.vision_decoder(dec_all)  # [1, N, out_chans]

                tgt = patchified[b:b+1].index_select(1, mask_perm)
                norm_target = self.patch_norm(tgt.detach())
                pred_masked = pred[:, -mask_perm.numel():, :]
                loss_sum = loss_sum + self.criterion(pred_masked, norm_target)
                denom = denom + mask_perm.numel() * self.out_chans

                if masked_patch_tokens is not None and mask_perm.numel() > 0:
                    masked_patch_tokens[b, mask_perm] = 0

                total_masked_patches += mask_perm.numel()

                if return_image:
                    pred_full = torch.zeros_like(patchified[b:b+1])
                    pred_full[:, shuffle_idx, :] = pred
                    recon_patches[b:b+1] = pred_full

            if denom > 0:
                mse_loss = loss_sum / denom
            else:
                mse_loss = image.new_tensor(0.0)

            if return_image and recon_patches is not None:
                patch_mean = patchified.mean(dim=-1, keepdim=True)
                patch_std = (
                    patchified.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6
                )
                recon_patches = recon_patches * patch_std + patch_mean
                recon_image = unpatchify(
                    recon_patches,
                    self.grid_size,
                    self.patch_size,
                    self.encoder_in_channels,
                )
            if masked_patch_tokens is not None:
                masked_image = unpatchify(
                    masked_patch_tokens,
                    self.grid_size,
                    self.patch_size,
                    self.encoder_in_channels,
                )
                if return_image:
                    arr = masked_image[0, 0].detach().float().cpu().numpy()
                    nib.save(nib.Nifti1Image(arr, np.eye(4)), "./masked_sample0.nii.gz")
                    if total_patches > 0:
                        print(f"[MAE] masked patch fraction: {total_masked_patches / total_patches:.4f}")
                    exit(0)

        ### ------------------CLIP loss 1: organ-name contrastive (TotalSegmentator)------------------ ###
        clip_loss_organ = torch.tensor(
            0.0, device=image.device, dtype=image.dtype
        )

        num_clip_entries = clip_token_masks.size(0)
        if self.clip_loss_weight > 0 and num_clip_entries > 0:
            # build text batch from pre-tokenized buffers
            text_input_ids = self.id2_input_ids[clip_organ_ids]        # [N_clip, L]
            text_attn_mask = self.id2_attention_mask[clip_organ_ids]   # [N_clip, L]
            text_emb = self.encode_text(text_input_ids, text_attn_mask)  # [B_valid, D]

            # encode the full volume once and keep only spatial tokens
            tokens_full = self.vision_encoder.patch_embedding(image)  # [B, N, H_enc]
            enc_all_full, _ = self.vision_encoder(tokens_full)        # [B, N, H_enc']
            if self.classification:
                enc_all_full = enc_all_full[:, 1:, :]

            # convert the voxel-level organ mask into patch-space indicators
            organ_token_mask = F.avg_pool3d(
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
                organ_tokens = vis_tokens_all[idx][token_mask]
                if organ_tokens.shape[0] == 0:
                    # fallback: average full grid if mask collapsed (rare but safe)
                    organ_tokens = vis_tokens_all[idx].mean(dim=0, keepdim=True)

                query_token = organ_query_tokens[idx:idx+1]  # [1,1,H_enc']
                tokens_plus_query = torch.cat(
                    [organ_tokens.unsqueeze(0), query_token], dim=1
                )  # [1, N_org+1, H_enc']

                # tiny self-attention over the selected tokens + query token
                attn_out, _ = self.organ_attn(
                    tokens_plus_query, tokens_plus_query, tokens_plus_query
                )
                tokens_plus_query = self.organ_attn_norm(tokens_plus_query + attn_out)

                ffn_out = self.organ_ffn(tokens_plus_query)
                tokens_plus_query = self.organ_ffn_norm(tokens_plus_query + ffn_out)

                pooled_embeds.append(tokens_plus_query[:, -1, :].squeeze(0))  # query slot

            vis_emb = torch.stack(pooled_embeds, dim=0)          # [N_clip, H_enc']
            vis_emb = self.mm_vision_proj(vis_emb)               # [B_valid, D]
            vis_emb = F.normalize(vis_emb, dim=-1)               # [B_valid, D]

            logit_scale = self.logit_scale.exp()

            # Gather features across all GPUs for organ CLIP
            if self.gather_loss:
                from LaMed.src.utils.dist_utils import gather_features
                all_vis_emb, all_text_emb = gather_features(vis_emb, text_emb)
                logits_per_image = logit_scale * all_vis_emb @ all_text_emb.T
                logits_per_text = logits_per_image.T
                B_valid = all_vis_emb.size(0)
            else:
                logits_per_image = logit_scale * vis_emb @ text_emb.T
                logits_per_text = logit_scale * text_emb @ vis_emb.T
                B_valid = vis_emb.size(0)
            
            targets = torch.arange(B_valid, device=image.device)

            loss_i2t = F.cross_entropy(logits_per_image, targets)
            loss_t2i = F.cross_entropy(logits_per_text, targets)
            clip_loss_organ = 0.5 * (loss_i2t + loss_t2i)

        ### ------------------CLIP loss 2: caption contrastive (CT-RATE captions)------------------ ###
        clip_loss_caption = torch.tensor(
            0.0, device=image.device, dtype=image.dtype
        )
        
        # Skip computation if weight is 0 (backward compatibility)
        if (
            self.caption_clip_loss_weight > 0
            and input_ids is not None
            and attention_mask is not None
        ):
            # Encode full image (not organ-only) for caption CLIP
            # Similar to M3DCLIP.encode_image: use CLS token
            image_feats, _ = self.vision_encoder(image)  # [B, N+1, H_enc] if classification=True (includes CLS)
            image_features = image_feats[:, 0, :]  # [B, H_enc] - CLS token
            image_features = self.mm_vision_proj(image_features)  # [B, D_clip]
            image_features = F.normalize(image_features, dim=-1)  # [B, D_clip]
            
            # Encode caption text
            text_features = self.encode_caption_text(input_ids, attention_mask)  # [B, D_clip]
            
            # Compute CLIP loss (similar to M3DCLIP)
            if self.gather_loss:
                from LaMed.src.utils.dist_utils import gather_features
                all_image_features, all_text_features = gather_features(image_features, text_features)
                if self.local_loss:
                    logits_per_image = self.logit_scale.exp() * image_features @ all_text_features.T
                    logits_per_text = self.logit_scale.exp() * text_features @ all_image_features.T
                else:
                    logits_per_image = self.logit_scale.exp() * all_image_features @ all_text_features.T
                    logits_per_text = logits_per_image.T
            else:
                logits_per_image = self.logit_scale.exp() * image_features @ text_features.T
                logits_per_text = self.logit_scale.exp() * text_features @ image_features.T
            
            # Use provided labels if available, otherwise create diagonal labels
            if labels is None:
                batch_size = image_features.size(0)
                if self.gather_loss and not self.local_loss:
                    if torch.distributed.is_initialized():
                        world_size = torch.distributed.get_world_size()
                        batch_size *= world_size
                labels = torch.arange(batch_size, device=image.device, dtype=torch.long)
            
            loss_i2t_caption = F.cross_entropy(logits_per_image, labels)
            loss_t2i_caption = F.cross_entropy(logits_per_text, labels)
            clip_loss_caption = 0.5 * (loss_i2t_caption + loss_t2i_caption)

        total_loss = (
            self.mae_loss_weight * mse_loss
            + self.clip_loss_weight * clip_loss_organ
            + self.caption_clip_loss_weight * clip_loss_caption
            + self.regular_mae_loss_weight * regular_mae_loss
        )
            
        if not return_image:
            return {
                "loss": total_loss,
                "mae_loss": mse_loss,
                "regular_mae_loss": regular_mae_loss,
                "clip_loss_organ": clip_loss_organ,
                "clip_loss_caption": clip_loss_caption,
            }

        return {
            "loss": total_loss,
            "orig_image": orig_image.detach(),
            "masked_image": masked_image.detach() if masked_image is not None else None,
            "recon_image": recon_image.detach(),
            "mae_masks": mae_masks.detach(),         # [B,1,D,H,W]
            "clip_masks": mae_masks.detach(),         # [B,1,D,H,W]
        }