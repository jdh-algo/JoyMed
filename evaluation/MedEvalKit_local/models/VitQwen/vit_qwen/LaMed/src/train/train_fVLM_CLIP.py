from typing import Optional
import transformers
from transformers import Trainer
from dataclasses import dataclass, field
from LaMed.src.dataset.multi_dataset import CTRATEDataset_w_Seg_fvlm
from LaMed.src.model.fVLM_CLIP import fVLM, fVLMConfig
from transformers import BertTokenizer
import torch
import numpy as np
from safetensors.torch import load_file
import os
from transformers import logging


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    language_model_name_or_path: str = field(default="./LaMed/pretrained_model/pubmedbert/")
    #language_model_name_or_path: str = field(default="bert-base-uncased")

    gather_loss: bool = field(default=True, metadata={"help": "Gather all distributed batch data of multiple GPUs and calculate contrastive loss together."})
    local_loss: bool = field(default=False)

    pretrained_model: str = field(default=None)
    encoder_in_channels: int = field(default=1)
    img_size: tuple = field(default=(128, 256, 256))
    patch_size: tuple = field(default=(8, 16, 16))

    encoder_hidden_size: int = field(default=768)
    decoder_hidden_size: int = field(default=768)
    mlp_dim: int = field(default=3072)
    num_layers: int = field(default=12)
    num_heads: int = field(default=12)
    pos_embed: str = field(default="perceptron")
    dropout_rate: float = field(default=0.0)
    spatial_dims: int = field(default=3)
    classification: bool = field(default=True)
    text_max_length: int = field(default=128)
    num_mask_organs: int = field(default=1)
    clip_loss_weight: float = field(default=1.0)   # keep organ-level CLIP


@dataclass
class DataArguments:
    data_root: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-CAP/M3D_Cap_nii", metadata={"help": "Root directory for all data."})
    # caption data
    cap_data_path: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-CAP/01_process_jdh/train_m3d_cap_caption_revised_filtered_93k_1117.jsonl", metadata={"help": "Path to caption data."})
    cap_data_path_CT_rate: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/01_processed_jdh/train_ct_rate_caption_full_47k_251114.jsonl", metadata={"help": "Path to CT-RATE caption data."})
    # cap_data_path_CT_rate_val: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/01_processed_jdh/val_ct_rate_caption.jsonl", metadata={"help": "Path to CT-RATE val caption data."})
    max_length: int = field(default=512)
    use_CT_rate_data: bool = field(default=True)
    use_M3D_data: bool = field(default=False)
    ct_rate_mask_root: str = field(default="/mnt/workspace/offline/xiehuidong.6688/fvlm/fvlm-main/data/merged_train_masks")
    ct_rate_conc_info: str = field(default="/mnt/workspace/offline/xiehuidong.6688/fvlm/des/conc_info.json")
    ct_rate_desc_info: str = field(default="/mnt/workspace/offline/xiehuidong.6688/fvlm/des/desc_info.json")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)

    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False

    # config in bash file
    bf16: bool = True
    output_dir: str = "./LaMed/output/CLIP"
    num_train_epochs: int = 100
    per_device_train_batch_size: int = 32 #32
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04 # 0.04
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 1
    learning_rate: float = 1e-4 #1e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    #logging_steps: float = 0.001 # 0.001
    logging_steps: int = 10
    gradient_checkpointing: bool = False # train fast
    dataloader_pin_memory: bool = True # fast
    dataloader_num_workers: int = 8
    report_to: str = "tensorboard"
    disable_tqdm = False


def compute_metrics(eval_pred):
    """
    Evaluation is not meaningful for CLIP-style contrast here; return a safe
    placeholder accuracy computed on overlapping length to avoid shape errors.
    """
    preds = np.array(eval_pred.predictions)
    labels = np.array(eval_pred.label_ids)
    preds_flat = preds.reshape(-1)
    labels_flat = labels.reshape(-1)
    m = min(preds_flat.shape[0], labels_flat.shape[0])
    if m == 0:
        return {}
    acc = (preds_flat[:m] == labels_flat[:m]).mean()
    return {"accuracy": float(acc)}

def preprocess_logits_for_metrics(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    return preds

@dataclass
class DataCollator:
    def __init__(self, gather_all):
        self.gather_all = gather_all
        if self.gather_all and torch.distributed.is_available() and torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
        else:
            self.world_size = 1

    def __call__(self, batch: list) -> dict:
        images, segs, texts, input_ids, attention_mask, organ_input_ids, organ_attention_mask, organ_ids, organ_normal_flags = tuple(
            [b[key] for b in batch]
            for key in (
                "image",
                "seg",
                "text",
                "input_id",
                "attention_mask",
                "organ_input_ids",
                "organ_attention_mask",
                "organ_ids",
                "organ_normal_flags",
            )
        )

        images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
        segs = torch.cat([_.unsqueeze(0) for _ in segs], dim=0)
        input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
        attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

        organ_input_ids_list = []
        organ_attention_mask_list = []
        organ_ids_list = []
        organ_batch_indices_list = []
        organ_normal_flags_list = []
        for b_idx, (org_inp, org_mask, org_ids_tensor) in enumerate(
            zip(organ_input_ids, organ_attention_mask, organ_ids)
        ):
            if org_inp.numel() == 0:
                continue
            organ_input_ids_list.append(org_inp)
            organ_attention_mask_list.append(org_mask)
            organ_ids_list.append(org_ids_tensor)
            organ_normal_flags_list.append(organ_normal_flags[b_idx])
            organ_batch_indices_list.append(
                torch.full(
                    (org_ids_tensor.shape[0],), b_idx, dtype=torch.long
                )
            )

        if organ_input_ids_list:
            organ_input_ids = torch.cat(organ_input_ids_list, dim=0)
            organ_attention_mask = torch.cat(organ_attention_mask_list, dim=0)
            organ_ids = torch.cat(organ_ids_list, dim=0)
            organ_batch_indices = torch.cat(organ_batch_indices_list, dim=0)
            organ_normal_flags = torch.cat(organ_normal_flags_list, dim=0)
        else:
            token_len = input_ids.shape[-1]
            organ_input_ids = torch.zeros((0, token_len), dtype=torch.long)
            organ_attention_mask = torch.zeros((0, token_len), dtype=torch.long)
            organ_ids = torch.zeros((0,), dtype=torch.long)
            organ_batch_indices = torch.zeros((0,), dtype=torch.long)
            organ_normal_flags = torch.zeros((0,), dtype=torch.bool)

        batch_size = images.shape[0]
        if self.gather_all:
            world_size = torch.distributed.get_world_size()
            batch_size *= world_size

        labels = torch.arange(batch_size, device=images.device, dtype=torch.long)

        return_dict = dict(
            image=images,
            seg=segs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            organ_input_ids=organ_input_ids,
            organ_attention_mask=organ_attention_mask,
            organ_batch_indices=organ_batch_indices,
            organ_ids=organ_ids,
            organ_normal_flags=organ_normal_flags,
            labels=labels,
        )

        return return_dict


def main():
    logging.set_verbosity_info()  # or logging.set_verbosity_debug()
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = BertTokenizer.from_pretrained(model_args.language_model_name_or_path)

    config_kwargs = vars(model_args).copy()
    # Backward-friendly aliases
    if "in_channels" in config_kwargs and "encoder_in_channels" not in config_kwargs:
        config_kwargs["encoder_in_channels"] = config_kwargs["in_channels"]
    if "hidden_size" in config_kwargs and "encoder_hidden_size" not in config_kwargs:
        config_kwargs["encoder_hidden_size"] = config_kwargs["hidden_size"]
    if "hidden_size" in config_kwargs and "decoder_hidden_size" not in config_kwargs:
        config_kwargs["decoder_hidden_size"] = config_kwargs["hidden_size"]

    config = fVLMConfig(**config_kwargs)
    # Ensure dropout is present for the ViT backbone
    if not hasattr(config, "dropout_rate"):
        config.dropout_rate = getattr(model_args, "dropout_rate", 0.0)

    model = fVLM(config)

    if model_args.pretrained_model:
        # ckpt = torch.load(model_args.pretrained_model)
        if model_args.pretrained_model.endswith(".safetensors"):
            ckpt = load_file(model_args.pretrained_model)
            model.load_state_dict(ckpt, strict=False)
            print(f"Loaded pretrained model from {model_args.pretrained_model}")
        else:
            # assume pytorch .bin or .pth
            ckpt = torch.load(model_args.pretrained_model, map_location="cpu")
            # Remap keys: add "vision_encoder." prefix
            vision_ckpt = {}
            for k, v in ckpt.items():
                vision_ckpt[f"vision_encoder.{k}"] = v
            ckpt = vision_ckpt
            
            model.load_state_dict(ckpt, strict=False)
            print(f"Loaded pretrained model from {model_args.pretrained_model}")
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            # print("Missing keys:", missing)
            # print("Unexpected keys:", unexpected)
            # exit(0)

    train_dataset = CTRATEDataset_w_Seg_fvlm(data_args, model_args, tokenizer, mode='train')
    eval_dataset = CTRATEDataset_w_Seg_fvlm(data_args, model_args, tokenizer, mode='validation')

    if model_args.gather_loss and not model_args.local_loss:
        gather_all = True
    else:
        gather_all = False
    data_collator = DataCollator(gather_all)

    trainer = Trainer(
                        model=model,
                        args=training_args,
                        data_collator=data_collator,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        compute_metrics=compute_metrics,
                        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                      )

    # Check if checkpoint folder is empty
    resume_from_checkpoint = False
    if os.path.exists(training_args.output_dir):
        # Check for checkpoint directories (checkpoint-* pattern)
        checkpoint_dirs = [d for d in os.listdir(training_args.output_dir) 
                          if os.path.isdir(os.path.join(training_args.output_dir, d)) 
                          and d.startswith('checkpoint-')]
        # Also check for checkpoint files directly in output_dir
        checkpoint_files = [f for f in os.listdir(training_args.output_dir) 
                           if os.path.isfile(os.path.join(training_args.output_dir, f)) 
                           and (f.endswith('.pt') or f.endswith('.pth') or f.endswith('.bin') 
                                or f.endswith('.safetensors') or f == 'pytorch_model.bin' 
                                or f == 'model.safetensors' or f == 'training_state.bin')]
        
        if checkpoint_dirs or checkpoint_files:
            resume_from_checkpoint = True
            print(f"Found checkpoints in {training_args.output_dir}, resuming training...")
        else:
            print(f"No checkpoints found in {training_args.output_dir}, starting fresh training...")
    else:
        print(f"Output directory {training_args.output_dir} does not exist, starting fresh training...")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_state()
    model.config.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(training_args.output_dir, 'model_params.bin'))


if __name__ == "__main__":
    main()
