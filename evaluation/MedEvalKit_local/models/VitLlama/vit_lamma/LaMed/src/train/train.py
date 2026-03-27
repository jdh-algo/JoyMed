import os
import logging
from typing import Optional, List, Dict
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from dataclasses import dataclass, field
from LaMed.src.dataset.multi_dataset import UniDatasets, CapDataset, TextDatasets, VQADataset
#from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM
from LaMed.src.model.language_model import LamedLlamaForCausalLM
from LaMed.src.train.lamed_trainer import LaMedTrainer

from transformers import logging
from collections import defaultdict

import torch.nn.functional as F

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="microsoft/Phi-3-mini-4k-instruct", metadata={"help": "Path to the LLM or MLLM."})
    model_type: Optional[str] = field(default=None, metadata={"help": "llama2, phi3"})

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(default=False, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."})

    # image
    image_channel: int = field(default=1)
    img_size: tuple = field(default=(128, 256, 256))
    patch_size: tuple = field(default=(8, 16, 16))

    # vision
    vision_tower: Optional[str] = field(default="vit3d") # None, "vit3d"
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default=None, metadata={"help": "Path to pretrained model for ViT."})
    freeze_vision_tower: bool = field(default=False)

    # projector
    mm_projector_type: Optional[str] = field(default='spp', metadata={"help": "spp"})
    proj_layer_type: str = field(default="mlp", metadata={"help": "Type of layer in projector. options: [linear, mlp]."})
    proj_layer_num: int = field(default=2, metadata={"help": "Number of layers in projector."})
    proj_pooling_type: str = field(default="spatial", metadata={"help": "Type of pooling in projector. options: [spatial, sequence]."})
    proj_pooling_size: int = field(default=2, metadata={"help": "Size of pooling in projector."})

    # segvol
    segmentation_module: str = field(default=None, metadata={"help": "segvol"})
    pretrain_seg_module: str = field(default=None, metadata={"help": "Pretrained segvol model."})



@dataclass
class DataArguments:
    data_root: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-CAP/M3D_Cap_nii", metadata={"help": "Root directory for all data."})

    # caption data
    cap_data_path: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-CAP/01_process_jdh/train_m3d_cap_caption_revised_filtered_93k_1117.jsonl", metadata={"help": "Path to caption data."})
    cap_data_path_CT_rate: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/01_processed_jdh/train_ct_rate_caption_full_47k_251114.jsonl", metadata={"help": "Path to CT-RATE caption data."})
    # VQA data
    vqa_data_train_path: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/01_processed_jdh/train_ct_rate_vqa_sampled_100k_20251106.jsonl")
    vqa_data_val_path: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/CT-RATE/dataset/01_processed_jdh/train_ct_rate_vqa_sampled_100k_20251106_swift.jsonl")
    # vqa_data_train_path: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-VQA/M3D_VQA_train.csv", metadata={"help": "Path to training VQA data."})
    # vqa_data_val_path: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-VQA/M3D_VQA_val.csv", metadata={"help": "Path to validation VQA data."})
    # vqa_data_test_path: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-VQA/M3D_VQA_test.csv", metadata={"help": "Path to testing VQA data."})

    vqa_yn_data_train_path: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-VQA/M3D_VQA_yn_train.csv", metadata={"help": "Path to training VQA Yes or No data."})

    # positioning & segmentation data
    seg_data_path: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-Seg/M3D_Seg/", metadata={"help": "Path to segmentation data."})
    refseg_data_train_path: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-RefSeg/M3D_RefSeg/M3D_RefSeg.csv", metadata={"help": "Path to refering segmentation data."})
    refseg_data_test_path: str = field(default="/mnt/workspace/offline/shared_data/Medical_Data_3D/M3D-RefSeg/M3D_RefSeg_test.csv", metadata={"help": "Path to refering segmentation data."})

    use_CT_rate_data: bool = field(default=True)
    use_M3D_data: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512, #512
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    seed: int = 42
    ddp_backend: str = "nccl"
    ddp_timeout: int = 128000
    ddp_find_unused_parameters: bool = False
    optim: str = field(default="adamw_torch")

    # This is set up to facilitate debugging, pls config these in bash file in training.
    bf16: bool = True
    output_dir: str = "./LaMed/output/LaMed-pretrain-test"
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10 # 0.001
    gradient_checkpointing: bool = False # train fast
    dataloader_pin_memory: bool = True # fast
    dataloader_num_workers: int = 0
    report_to: str = "tensorboard"


def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    labels = labels_ids[:, 1:]
    preds = pred_ids[:, :-1]

    labels_flatten = labels.reshape(-1)
    preds_flatten = preds.reshape(-1)
    valid_indices = np.where(labels_flatten != -100)
    filtered_preds = preds_flatten[valid_indices]
    filtered_labels = labels_flatten[valid_indices]
    acc_score = sum(filtered_preds==filtered_labels) / len(filtered_labels)

    return {"accuracy": acc_score}

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save projector and embed_tokens in pretrain
        keys_to_match = ['mm_projector', 'embed_tokens']

        weight_to_save = get_mm_projector_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa



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

# @dataclass
# class DataCollator:
#     def __init__(self, seg_enable):
#         self.seg_enable = seg_enable
#     def __call__(self, batch: list) -> dict:
#         if self.seg_enable:
#             images, input_ids, labels, attention_mask, segs = tuple(
#                 [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask', 'seg'))

#             images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
#             input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
#             labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
#             attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

#             for i, seg in enumerate(segs):
#                 if seg.sum() == 0:
#                     segs[i] = torch.zeros((1, 1, 128, 256, 256))
#                 else:
#                     segs[i] = seg.unsqueeze(0)
#             segs = torch.cat(segs, dim=0)

#             return_dict = dict(
#                 images=images,
#                 input_ids=input_ids,
#                 labels=labels,
#                 attention_mask=attention_mask,
#                 segs=segs,
#             )
#         else:
#             images, input_ids, labels, attention_mask = tuple(
#                 [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask'))

#             images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
#             input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
#             labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
#             attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

#             return_dict = dict(
#                 images=images,
#                 input_ids=input_ids,
#                 labels=labels,
#                 attention_mask=attention_mask,
#             )

#         return return_dict
        
@dataclass
class DataCollator:
    def __init__(self, seg_enable, pad_token_id):
        self.seg_enable = seg_enable
        self.pad_token_id = pad_token_id

    def _pad_1d(self, tensors, pad_value):
        """
        tensors: list of 1D tensors [seq_len]
        Returns: [batch_size, max_len] padded tensor
        """
        max_len = max(t.size(0) for t in tensors)
        padded = []
        for t in tensors:
            if t.size(0) == max_len:
                padded.append(t)
            else:
                pad_len = max_len - t.size(0)
                padded.append(F.pad(t, (0, pad_len), value=pad_value))
        return torch.stack(padded, dim=0)

    def __call__(self, batch: list) -> dict:
        if self.seg_enable:
            images, input_ids, labels, attention_mask, segs = tuple(
                [b[key] for b in batch]
                for key in ('image', 'input_id', 'label', 'attention_mask', 'seg')
            )
        else:
            images, input_ids, labels, attention_mask = tuple(
                [b[key] for b in batch]
                for key in ('image', 'input_id', 'label', 'attention_mask')
            )

        # images already have same shape due to Resize, just stack
        images = torch.cat([img.unsqueeze(0) for img in images], dim=0)

        # ---- pad sequence fields to max length in batch ----
        input_ids      = self._pad_1d(input_ids,      pad_value=self.pad_token_id)
        labels         = self._pad_1d(labels,         pad_value=-100)  # ignore index
        attention_mask = self._pad_1d(attention_mask, pad_value=0)     # 0 = padded

        if self.seg_enable:
            # keep your existing seg logic
            for i, seg in enumerate(segs):
                if seg.sum() == 0:
                    segs[i] = torch.zeros((1, 1, 128, 256, 256))
                else:
                    segs[i] = seg.unsqueeze(0)
            segs = torch.cat(segs, dim=0)

            return dict(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                segs=segs,
            )
        else:
            return dict(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

def main():
    logging.set_verbosity_info()  # or logging.set_verbosity_debug()
    
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank

    rank0_print("="*20 + " Tokenizer preparation " + "="*20)
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Define and add special tokens
    special_token = {"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]}
    tokenizer.add_special_tokens(
        special_token
    )
    tokenizer.add_tokens("[SEG]")

    # # Resize model token embeddings with padding for Tensor Core alignment
    # model.resize_token_embeddings(
    #     len(tokenizer),
    #     pad_to_multiple_of=64 
    # )

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if 'llama3' in model_args.model_type:
        tokenizer.eos_token_id = 128001
        tokenizer.pad_token = tokenizer.eos_token

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    model_args.vocab_size = len(tokenizer)
    rank0_print("seg_token_id: ", model_args.seg_token_id)
    rank0_print("vocab_size: ", model_args.vocab_size)

    rank0_print("="*20 + " Model preparation " + "="*20)
    if model_args.vision_tower is not None:
        if 'llama' in model_args.model_type:
            model = LamedLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elif 'phi3' in model_args.model_type:
            model = LamedPhi3ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir
                )
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )

    model.config.seg_token_id = model_args.seg_token_id
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    model.enable_input_require_grads()
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print("model_args.vision_tower ",model_args.vision_tower)
    
    # initialize vision and seg modules on LLM
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)
    if model_args.segmentation_module is not None:
        model.get_model().initialize_seg_modules(model_args=model_args)
    
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model_args.num_new_tokens = 4
    model.initialize_vision_tokenizer(model_args, tokenizer)

    if model_args.pretrain_mllm:
        ckpt = torch.load(model_args.pretrain_mllm, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        rank0_print("load pretrained MLLM weights.")

    if model_args.freeze_backbone:
        model.model.embed_tokens.weight.requires_grad = False

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
        rank0_print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)

        for n, p in model.named_parameters():
            if any(
                    [x in n for x in ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']]
            ):
                p.requires_grad = True

        model.print_trainable_parameters()

    # ckpt = torch.load("PATH/model_with_lora.bin", map_location="cpu")
    # model.load_state_dict(ckpt, strict=True)

    rank0_print("="*20 + " Dataset preparation " + "="*20)
    #data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model.get_model().mm_projector.proj_out_num
    data_args.max_length = training_args.model_max_length - data_args.proj_out_num
    rank0_print("vision tokens output from projector: ", data_args.proj_out_num)
    rank0_print("text max_length (excl.vision tokens): ", data_args.max_length)
    data_args.seg_enable = hasattr(model.get_model(), "seg_module")

    if model_args.tune_mm_mlp_adapter:
        train_dataset = TextDatasets(data_args, model_args, tokenizer, mode='train')
    else:
        train_dataset = UniDatasets(data_args, model_args, tokenizer, mode='train')

    eval_dataset = CapDataset(data_args, model_args, tokenizer, mode='validation')
    data_collator = DataCollator(data_args.seg_enable, tokenizer.pad_token_id)

    print("lora_enable ", training_args.lora_enable)
    print("freeze_backbone ", model_args.freeze_backbone)
    print("tune_mm_mlp_adapter ", model_args.tune_mm_mlp_adapter)
    print("freeze_vision_tower ", model_args.freeze_vision_tower)

    print("=== ALL MODEL PARAMETERS (Trainable & Frozen) ===")

    # Initialize counters
    groups = {
        "Vision Encoder": 0,
        "LLM (including embed_tokens)": 0,
        "3D Spatial Pooling Perceiver": 0,
        "Segmentation Module": 0,
        "Other": 0
    }
    group_trainable = defaultdict(int)
    group_frozen = defaultdict(int)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        is_trainable = param.requires_grad
    
        # Categorize
        if "vision_tower" in name:
            group = "Vision Encoder"
        elif "mm_projector" in name:
            group = "3D Spatial Pooling Perceiver"
        elif "seg_module" in name or "seg_projector" in name:
            group = "Segmentation Module"
        elif any(k in name for k in ["model.layers", "lm_head", "embed_tokens"]):
            group = "LLM (including embed_tokens)"
        else:
            group = "Other"
    
        groups[group] += param_count
        if is_trainable:
            group_trainable[group] += param_count
        else:
            group_frozen[group] += param_count
    
    # Print grouped summary
    print("\n" + "="*60)
    print("PARAMETER GROUP SUMMARY (in millions)")
    print("="*60)
    for group in groups:
        total_m = groups[group] / 1e6
        trainable_m = group_trainable[group] / 1e6
        frozen_m = group_frozen[group] / 1e6
        print(f"{group:30} | Total: {total_m:6.1f}M | Trainable: {trainable_m:6.1f}M | Frozen: {frozen_m:6.1f}M")
    
    total_all = sum(groups.values())
    total_trainable = sum(group_trainable.values())
    total_frozen = sum(group_frozen.values())
    
    print("-"*60)
    print(f"{'Overall TOTAL':30} | Total: {total_all / 1e6:6.1f}M | Trainable: {total_trainable / 1e6:6.1f}M | Frozen: {total_frozen / 1e6:6.1f}M")
    print("="*60)

    # total_trainable = 0
    # total_frozen = 0
    
    # for name, param in model.named_parameters():
    #     param_count = param.numel()
    #     status = "TRAINABLE" if param.requires_grad else "FROZEN"
    #     print(f"{name}: {param_count / 1e6:.1f}M ({status})")
        
    #     if param.requires_grad:
    #         total_trainable += param_count
    #     else:
    #         total_frozen += param_count
    
    # print(f"\nTotal TRAINABLE: {total_trainable / 1e6:.1f}M")
    # print(f"Total FROZEN: {total_frozen / 1e6:.1f}M")
    # print(f"Total PARAMETERS: {(total_trainable + total_frozen) / 1e6:.1f}M")
    # exit(0)

    rank0_print("="*20 + " Training " + "="*20)
    trainer = LaMedTrainer(
                            model=model,
                            args=training_args,
                            data_collator=data_collator,
                            train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            compute_metrics=compute_metrics,
                            preprocess_logits_for_metrics=preprocess_logits_for_metrics
                      )

    trainer.train()
    trainer.save_state()
    model.config.use_cache = True

    rank0_print("="*20 + " Save model " + "="*20)
    if training_args.lora_enable:
        state_dict_with_lora = model.state_dict()
        torch.save(state_dict_with_lora, os.path.join(training_args.output_dir, 'model_with_lora.bin'))
        trainer.model.config.save_pretrained(output_dir)

    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)




if __name__ == "__main__":
    main()
