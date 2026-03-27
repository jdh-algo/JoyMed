from typing import Optional
import transformers
from transformers import Trainer
from dataclasses import dataclass, field
from LaMed.src.dataset.multi_dataset import ITRDataset
from LaMed.src.model.CLIP import M3DCLIP, M3DCLIPConfig
from transformers import BertTokenizer
import torch
from safetensors.torch import load_file
import os

from monai.transforms import Resize
from transformers import logging


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    language_model_name_or_path: str = field(default="./LaMed/pretrained_model/pubmedbert/")
    #language_model_name_or_path: str = field(default="bert-base-uncased")

    gather_loss: bool = field(default=True, metadata={"help": "Gather all distributed batch data of multiple GPUs and calculate contrastive loss together."})
    local_loss: bool = field(default=False)

    pretrained_model: str = field(default=None)
    in_channels: int = field(default=1)
    img_size: tuple = field(default=(128, 256, 256))
    patch_size: tuple = field(default=(8, 16, 16))

    hidden_size: int = field(default=768)
    mlp_dim: int = field(default=3072)
    num_layers: int = field(default=12)
    num_heads: int = field(default=12)
    pos_embed: str = field(default="perceptron")
    dropout_rate: float = field(default=0.0)
    spatial_dims: int = field(default=3)
    max_text_len: int = field(default=128)
    vocab_size: int = field(default=30522)


@dataclass
class DataArguments:
    data_root: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-CAP/M3D_Cap_nii", metadata={"help": "Root directory for all data."})
    # caption data
    cap_data_path: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-CAP/01_process_jdh/train_m3d_cap_caption_revised_filtered_93k_1117.jsonl", metadata={"help": "Path to caption data."})
    cap_data_path_CT_rate: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/01_processed_jdh/train_ct_rate_caption_full_47k_251114.jsonl", metadata={"help": "Path to CT-RATE caption data."})
    # cap_data_path_CT_rate_val: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/01_processed_jdh/val_ct_rate_caption.jsonl", metadata={"help": "Path to CT-RATE val caption data."})
    max_length: int = field(default=512)
    use_CT_rate_data: bool = field(default=False)
    use_M3D_data: bool = field(default=True)


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
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    correct = (preds == labels).sum()
    total = labels.size
    acc = correct / total
    return {"accuracy": acc}

def preprocess_logits_for_metrics(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    return preds

@dataclass
class DataCollator:
    def __init__(self, gather_all):
        self.gather_all = gather_all

    def __call__(self, batch: list) -> dict:
        images, texts, input_ids, attention_mask = tuple(
            [b[key] for b in batch] for key in ('image', 'text', 'input_id', 'attention_mask'))
        
        images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
        input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
        attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

        batch_size = images.shape[0]
        if self.gather_all:
            world_size = torch.distributed.get_world_size()
            batch_size *= world_size

        labels = torch.arange(batch_size, device=images.device, dtype=torch.long)

        return_dict = dict(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,

        )

        return return_dict


def main():
    logging.set_verbosity_info()  # or logging.set_verbosity_debug()
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = BertTokenizer.from_pretrained(model_args.language_model_name_or_path)

    config = M3DCLIPConfig.from_dict(vars(model_args))
    model = M3DCLIP(config)

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

    train_dataset = ITRDataset(data_args, model_args, tokenizer, mode='train')
    eval_dataset = ITRDataset(data_args, model_args, tokenizer, mode='validation')

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

    # if you want to resume your training, pls set the checkpoint in trainer.train(resume_from_checkpoint="")
    trainer.train()

    trainer.save_state()
    model.config.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(training_args.output_dir, 'model_params.bin'))


if __name__ == "__main__":
    main()
