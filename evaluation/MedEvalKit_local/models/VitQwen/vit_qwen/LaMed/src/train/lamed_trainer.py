import os
import torch
from transformers import Trainer
from transformers.utils import logging, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from typing import Optional

logger = logging.get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"

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
    

class LaMedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track step count for periodic detailed LR debugging
        self._last_detailed_lr_log_step = -1
        # Track actual LR usage during training steps
        self._training_step_lr_verification = {}
        self._verification_step_count = 0
        # Initialize optimizer step hook tracking (will be set if hook is used)
        self._original_optimizer_step = None
        self._optimizer_step_count = 0
        self._optimizer_step_lrs = []
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple of
        (optimizer_cls, optimizer_kwargs) to the Trainer constructor or override this method in a subclass.
        """
        if self.optimizer is not None:
            return self.optimizer

        # If vision_learning_rate is not set, use default behavior
        if getattr(self.args, "vision_learning_rate", None) is None:
            return super().create_optimizer()
        
        # If vision_learning_rate equals learning_rate, no need for separate groups
        if abs(getattr(self.args, "vision_learning_rate", 0) - self.args.learning_rate) < 1e-10:
            return super().create_optimizer()

        opt_model = self.model
        
        # Check if vision_tower has any trainable parameters
        # If vision_tower is frozen (no trainable params), use default optimizer creation
        has_trainable_vision_params = any(
            "vision_tower" in n and p.requires_grad 
            for n, p in opt_model.named_parameters()
        )
        
        if not has_trainable_vision_params:
            # Vision tower is frozen, no need for separate learning rates
            return super().create_optimizer()
        
        # Robustly identify parameters that should NOT have weight decay
        # This uses HF's internal logic to exclude biases and normalization layers (LayerNorm, RMSNorm, etc.)
        decay_parameters = self.get_decay_parameter_names(opt_model)
        
        optimizer_grouped_parameters = [
            # 1. Vision Encoder (with weight decay)
            {
                "params": [
                    p for n, p in opt_model.named_parameters() 
                    if "vision_tower" in n and n in decay_parameters and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.vision_learning_rate,
            },
            # 2. Vision Encoder (no weight decay)
            {
                "params": [
                    p for n, p in opt_model.named_parameters() 
                    if "vision_tower" in n and n not in decay_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": self.args.vision_learning_rate,
            },
            # 3. Everything Else (with weight decay)
            {
                "params": [
                    p for n, p in opt_model.named_parameters() 
                    if "vision_tower" not in n and n in decay_parameters and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            # 4. Everything Else (no weight decay)
            {
                "params": [
                    p for n, p in opt_model.named_parameters() 
                    if "vision_tower" not in n and n not in decay_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate,
            },
        ]
        
        # Filter out empty parameter groups to avoid scheduler issues
        optimizer_grouped_parameters = [group for group in optimizer_grouped_parameters if len(group["params"]) > 0]

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
        # Store group type information for later use in logging
        # We identify groups by their initial learning rate
        for group in self.optimizer.param_groups:
            if abs(group['lr'] - self.args.vision_learning_rate) < 1e-8:
                group['_group_type'] = 'vision'
            elif abs(group['lr'] - self.args.learning_rate) < 1e-8:
                group['_group_type'] = 'other'
            else:
                group['_group_type'] = 'unknown'
        
        # Note: optimizer step hook tracking variables are initialized in __init__
        # The hook will be set up in training_step() after scheduler is created

        # Verification print - use is_world_process_zero for reliability
        if self.is_world_process_zero():
            print("\n" + "="*60)
            print("OPTIMIZER PARAMETER GROUPS VERIFICATION")
            print("="*60)
            group_names = [
                "Vision Encoder (Decay)", 
                "Vision Encoder (No Decay)", 
                "Others (Decay)", 
                "Others (No Decay)"
            ]
            
            # Create param name mapping for verification
            param_to_name_verify = {p: n for n, p in opt_model.named_parameters()}
            if hasattr(opt_model, 'get_base_model'):
                base_model = opt_model.get_base_model()
                for n, p in base_model.named_parameters():
                    if p not in param_to_name_verify:
                        param_to_name_verify[p] = n
            
            for i, group in enumerate(self.optimizer.param_groups):
                num_params = sum(p.numel() for p in group["params"])
                # Check if this is actually a vision group
                is_vision = any(
                    "vision_tower" in param_to_name_verify.get(p, "").lower()
                    for p in group["params"][:min(5, len(group["params"]))]  # Check first few
                )
                actual_type = "VISION" if is_vision else "OTHER"
                print(f"Group {i} [{group_names[i]} - Actually: {actual_type}]:")
                print(f"  - Parameters: {num_params / 1e6:.2f}M")
                print(f"  - Learning Rate: {group['lr']}")
                print(f"  - Weight Decay: {group['weight_decay']}")
                if group["params"]:
                    sample_name = param_to_name_verify.get(group["params"][0], "UNKNOWN")
                    print(f"  - Sample param: {sample_name[:80]}...")
            print("="*60 + "\n")

        return self.optimizer
    
    def training_step(self, model, inputs):
        """
        Override training step to verify learning rates are actually being used during optimization.
        """
        # Hook optimizer.step() after scheduler is created (only once, and only if not already hooked)
        if (hasattr(self, 'optimizer') and self.optimizer is not None and 
            hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None and
            getattr(self.args, "vision_learning_rate", None) is not None and
            self._original_optimizer_step is None):
            self._hook_optimizer_step()
        
        # Call parent training step
        loss = super().training_step(model, inputs)
        
        # Periodically verify optimizer is using correct LRs (every 20 steps)
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            if getattr(self.args, "vision_learning_rate", None) is not None:
                current_step = self.state.global_step if hasattr(self, 'state') and hasattr(self.state, 'global_step') else 0
                
                # Verify every 20 steps
                if current_step > 0 and current_step % 20 == 0:
                    self._verify_optimizer_lrs(current_step)
        
        return loss
    
    def _hook_optimizer_step(self):
        """
        Add a hook to optimizer.step() to verify different learning rates are used.
        This must be called AFTER scheduler is created to avoid conflicts.
        The scheduler wraps optimizer.step(), so we need to wrap the already-wrapped version.
        """
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            return
        
        # Store the current step method (which may already be wrapped by scheduler)
        import types
        self._original_optimizer_step = self.optimizer.step
        
        # Create a wrapper that preserves method binding
        # Check if original step accepts closure argument
        import inspect
        step_sig = inspect.signature(self._original_optimizer_step)
        accepts_closure = 'closure' in step_sig.parameters
        
        def wrapped_step(closure=None):
            # Get LRs before step (these are what will be used)
            lrs_before = [group['lr'] for group in self.optimizer.param_groups]
            
            # Call original step (which may be scheduler-wrapped or DeepSpeed-wrapped)
            # DeepSpeed optimizer doesn't accept closure argument
            if accepts_closure:
                result = self._original_optimizer_step(closure)
            else:
                result = self._original_optimizer_step()
            
            # Track this step
            self._optimizer_step_count += 1
            current_step = self.state.global_step if hasattr(self, 'state') and hasattr(self.state, 'global_step') else self._optimizer_step_count
            
            # Store LR info (only every 500 steps to avoid spam)
            if current_step % 500 == 0:
                # Identify groups by comparing current LR to expected base LRs
                # (accounting for warmup/decay - both will have same multiplier)
                vision_lrs = []
                other_lrs = []
                
                # Get all unique LR values and their counts
                lr_counts = {}
                for group in self.optimizer.param_groups:
                    current_lr = group.get('lr', 0)
                    if current_lr > 0:
                        lr_counts[current_lr] = lr_counts.get(current_lr, 0) + 1
                
                # Identify vision vs other by LR value
                # Vision LR should be ~5x smaller than other LR
                # During warmup, both are scaled by same factor, so ratio is preserved
                if hasattr(self.args, 'vision_learning_rate') and lr_counts:
                    unique_lrs = sorted(lr_counts.keys())
                    
                    # Simple heuristic: smaller LRs are vision, larger are other
                    # With 4 groups, we expect 2 vision groups (same LR) and 2 other groups (same LR)
                    if len(unique_lrs) >= 2:
                        # Take the two most common LR values
                        # The smaller one should be vision, larger should be other
                        smaller_lr = unique_lrs[0]
                        larger_lr = unique_lrs[-1]
                        
                        # Verify ratio is approximately 5x
                        if larger_lr > 0 and smaller_lr > 0:
                            ratio_check = larger_lr / smaller_lr
                            if 4.0 < ratio_check < 6.0:  # Allow some tolerance
                                # Assign based on LR value
                                for group in self.optimizer.param_groups:
                                    current_lr = group.get('lr', 0)
                                    if abs(current_lr - smaller_lr) < 1e-10:
                                        vision_lrs.append(current_lr)
                                    elif abs(current_lr - larger_lr) < 1e-10:
                                        other_lrs.append(current_lr)
                    elif len(unique_lrs) == 1:
                        # All groups have same LR (shouldn't happen, but handle it)
                        # Use midpoint to split
                        mid_point = (self.args.vision_learning_rate + self.args.learning_rate) / 2
                        for group in self.optimizer.param_groups:
                            current_lr = group.get('lr', 0)
                            if current_lr > 0:
                                if current_lr < mid_point:
                                    vision_lrs.append(current_lr)
                                else:
                                    other_lrs.append(current_lr)
                
                if vision_lrs and other_lrs:
                    vision_avg = sum(vision_lrs) / len(vision_lrs)
                    other_avg = sum(other_lrs) / len(other_lrs)
                    ratio = other_avg / vision_avg if vision_avg > 0 else 0
                    
                    self._optimizer_step_lrs.append({
                        'step': current_step,
                        'vision_lr': vision_avg,
                        'other_lr': other_avg,
                        'ratio': ratio,
                        'lrs_before': lrs_before
                    })
            
            return result
        
        # Replace step method - use MethodType to preserve binding
        self.optimizer.step = types.MethodType(wrapped_step, self.optimizer)
    
    def _verify_optimizer_lrs(self, step):
        """
        Verify that the optimizer is actually using different learning rates for different groups.
        This checks the optimizer's internal state to confirm LRs are being applied.
        """
        if not self.is_world_process_zero():
            return
        
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            return
        
        # Get current learning rates from optimizer (these are what will be used in next step)
        current_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        # Identify which groups are vision vs other
        vision_lrs_actual = []
        other_lrs_actual = []
        
        for i, group in enumerate(self.optimizer.param_groups):
            group_type = group.get('_group_type', 'unknown')
            if group_type == 'vision':
                vision_lrs_actual.append(group['lr'])
            elif group_type == 'other':
                other_lrs_actual.append(group['lr'])
        
        # Calculate averages
        vision_avg = sum(vision_lrs_actual) / len(vision_lrs_actual) if vision_lrs_actual else None
        other_avg = sum(other_lrs_actual) / len(other_lrs_actual) if other_lrs_actual else None
        
        # Verify ratio
        if vision_avg and other_avg and vision_avg > 0:
            actual_ratio = other_avg / vision_avg
            expected_ratio = self.args.learning_rate / self.args.vision_learning_rate
            
            # Store for later reporting
            if not hasattr(self, '_training_step_lr_verification'):
                self._training_step_lr_verification = {}
            self._training_step_lr_verification[step] = {
                'vision_lr': vision_avg,
                'other_lr': other_avg,
                'ratio': actual_ratio,
                'expected_ratio': expected_ratio,
                'all_lrs': current_lrs
            }
            
            # Print verification result immediately
            ratio_ok = "✓" if abs(actual_ratio - expected_ratio) < 0.1 else "✗"
            print(f"\n[Training Step {step} Verification] Vision LR: {vision_avg:.2e}, Other LR: {other_avg:.2e}, Ratio: {actual_ratio:.2f}x {ratio_ok}")
            
            # Print warning if ratio is wrong
            if abs(actual_ratio - expected_ratio) > 0.1:
                print(f"⚠️  WARNING: LR ratio mismatch during training!")
                print(f"   Expected ratio: {expected_ratio:.2f}x")
                print(f"   Actual ratio: {actual_ratio:.2f}x")
                print(f"   All group LRs: {current_lrs}\n")
    
    def log(self, logs: dict) -> None:
        """
        Log metrics and learning rates for each parameter group.
        Adds separate learning rate logging for vision encoder and other parameters.
        """
        # If vision_learning_rate is set and optimizer exists, add separate learning rates to logs
        if getattr(self.args, "vision_learning_rate", None) is not None and self.optimizer is not None:
            # Extract learning rates from parameter groups
            # Identify vision vs others by checking parameter names (robust to filtered empty groups)
            vision_lrs = []
            other_lrs = []
            
            # Use stored group type information (set during optimizer creation)
            # This is the most reliable way to identify groups
            param_to_name = {}
            for n, p in self.model.named_parameters():
                param_to_name[p] = n
            if hasattr(self.model, 'get_base_model'):
                base_model = self.model.get_base_model()
                for n, p in base_model.named_parameters():
                    if p not in param_to_name:
                        param_to_name[p] = n
            
            # Detailed group information for debugging
            group_details = []
            
            for group_idx, group in enumerate(self.optimizer.param_groups):
                if group["params"]:  # Only log if group has parameters
                    current_lr = group.get("lr", None)
                    if current_lr is not None:
                        # Use stored group type (set during optimizer creation)
                        group_type = group.get('_group_type', 'unknown')
                        
                        # If not stored, try to infer from current LR ratio
                        # (accounting for warmup/decay - both groups will have same multiplier)
                        if group_type == 'unknown':
                            # Get scheduler multiplier if available
                            if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                                # Try to get the multiplier by comparing to base LR
                                # During warmup, both will be scaled similarly
                                # We need to check which base LR this group started with
                                # Since we can't easily get that, try parameter name matching
                                sample_names = []
                                is_vision_by_name = False
                                for p in group["params"][:5]:
                                    param_name = param_to_name.get(p, "")
                                    if param_name != "UNKNOWN":
                                        sample_names.append(param_name)
                                        if "vision_tower" in param_name.lower():
                                            is_vision_by_name = True
                                            break
                                
                                if is_vision_by_name:
                                    group_type = 'vision'
                                else:
                                    # Fallback: assume it's 'other' if we can't determine
                                    group_type = 'other'
                            else:
                                # No scheduler yet, compare directly to base LRs
                                vision_diff = abs(current_lr - self.args.vision_learning_rate)
                                other_diff = abs(current_lr - self.args.learning_rate)
                                group_type = 'vision' if vision_diff < other_diff else 'other'
                        
                        is_vision_group = (group_type == 'vision')
                        
                        # Get sample parameter names for debugging
                        sample_names = []
                        for p in group["params"][:2]:
                            param_name = param_to_name.get(p, "UNKNOWN")
                            sample_names.append(param_name)
                        
                        num_params = sum(p.numel() for p in group["params"])
                        group_details.append({
                            "idx": group_idx,
                            "is_vision": is_vision_group,
                            "lr": current_lr,
                            "num_params": num_params,
                            "weight_decay": group.get("weight_decay", 0.0),
                            "sample_names": sample_names,
                            "group_type_stored": group.get('_group_type', 'not_stored')
                        })
                        
                        if is_vision_group:
                            vision_lrs.append(current_lr)
                        else:
                            other_lrs.append(current_lr)
            
            # Log the learning rates (use average if multiple groups have same type)
            if vision_lrs:
                logs["train/learning_rate_vision"] = sum(vision_lrs) / len(vision_lrs)
            if other_lrs:
                logs["train/learning_rate_others"] = sum(other_lrs) / len(other_lrs)
            
            # Periodic detailed debugging: print every logging_steps or every 100 steps, whichever is more frequent
            current_step = logs.get("step", None)
            if current_step is None and hasattr(self, 'state') and hasattr(self.state, 'global_step'):
                current_step = self.state.global_step
            if current_step is None:
                current_step = 0
            
            log_interval = max(1, getattr(self.args, "logging_steps", 10))
            detailed_log_interval = max(log_interval, 20)  # Print detailed info at least every 100 steps
            
            if current_step > 0 and current_step % detailed_log_interval == 0 and current_step != self._last_detailed_lr_log_step:
                if self.is_world_process_zero():
                    # Get scheduler info if available
                    scheduler_info = ""
                    if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                        if hasattr(self.lr_scheduler, 'get_last_lr'):
                            scheduler_lrs = self.lr_scheduler.get_last_lr()
                            scheduler_info = f" (Scheduler LRs: {[f'{lr:.2e}' for lr in scheduler_lrs]})"
                    
                    # Calculate actual LR ratio between vision and other groups
                    vision_avg_lr = sum(vision_lrs) / len(vision_lrs) if vision_lrs else None
                    other_avg_lr = sum(other_lrs) / len(other_lrs) if other_lrs else None
                    # Only calculate ratio if both values are valid
                    if vision_avg_lr is not None and vision_avg_lr > 0 and other_avg_lr is not None and other_avg_lr > 0:
                        actual_ratio = other_avg_lr / vision_avg_lr
                    else:
                        actual_ratio = None
                    expected_ratio = (self.args.learning_rate / self.args.vision_learning_rate) if (self.args.vision_learning_rate and self.args.vision_learning_rate > 0) else None
                    
                    # Format strings for display
                    vision_lr_str = f"{vision_avg_lr:.2e}" if vision_avg_lr is not None else "N/A"
                    other_lr_str = f"{other_avg_lr:.2e}" if other_avg_lr is not None else "N/A"
                    expected_ratio_str = f"{expected_ratio:.2f}x" if expected_ratio else "N/A"
                    
                    print("\n" + "="*70)
                    print(f"LEARNING RATE DEBUG - Step {current_step}{scheduler_info}")
                    print("="*70)
                    print(f"Expected base LRs: vision_lr={self.args.vision_learning_rate}, other_lr={self.args.learning_rate}")
                    print(f"Expected ratio (other/vision): {expected_ratio_str}")
                    print(f"Actual averages: vision_lr={vision_lr_str}, other_lr={other_lr_str}")
                    if actual_ratio and expected_ratio:
                        ratio_match = "✓" if abs(actual_ratio - expected_ratio) < 0.1 else f"✗ (expected {expected_ratio:.2f}x)"
                        print(f"Actual ratio (other/vision): {actual_ratio:.2f}x {ratio_match}")
                    print("-"*70)
                    print("Parameter Group Details:")
                    for detail in group_details:
                        group_type = "VISION" if detail["is_vision"] else "OTHER"
                        expected_base_lr = self.args.vision_learning_rate if detail["is_vision"] else self.args.learning_rate
                        # Check if current LR matches expected (accounting for warmup/decay)
                        # During warmup, LR will be smaller, so we just show the ratio
                        lr_ratio = detail["lr"] / expected_base_lr if expected_base_lr > 0 else 0
                        print(f"  Group {detail['idx']} [{group_type}]:")
                        print(f"    - Current LR: {detail['lr']:.2e} (base: {expected_base_lr:.2e}, warmup/decay ratio: {lr_ratio:.4f})")
                        print(f"    - Params: {detail['num_params'] / 1e6:.2f}M")
                        print(f"    - Weight Decay: {detail['weight_decay']}")
                        if detail.get("sample_names"):
                            print(f"    - Sample param names: {detail['sample_names']}")
                    # Show verification results from training steps
                    if hasattr(self, '_training_step_lr_verification') and self._training_step_lr_verification:
                        recent_steps = sorted(self._training_step_lr_verification.keys())[-3:]
                        print("Recent Training Step Verifications:")
                        for v_step in recent_steps:
                            v_data = self._training_step_lr_verification[v_step]
                            ratio_ok = "✓" if abs(v_data['ratio'] - v_data['expected_ratio']) < 0.1 else "✗"
                            print(f"  Step {v_step}: Vision={v_data['vision_lr']:.2e}, Other={v_data['other_lr']:.2e}, Ratio={v_data['ratio']:.2f}x {ratio_ok}")
                    
                    # Show optimizer.step() hook verification
                    if hasattr(self, '_optimizer_step_lrs') and self._optimizer_step_lrs:
                        recent_optimizer_steps = self._optimizer_step_lrs[-3:]
                        print("Optimizer.step() Hook Verifications (actual parameter updates):")
                        for opt_data in recent_optimizer_steps:
                            expected_ratio = self.args.learning_rate / self.args.vision_learning_rate
                            ratio_ok = "✓" if abs(opt_data['ratio'] - expected_ratio) < 0.1 else "✗"
                            print(f"  Step {opt_data['step']}: Vision={opt_data['vision_lr']:.2e}, Other={opt_data['other_lr']:.2e}, Ratio={opt_data['ratio']:.2f}x {ratio_ok}")
                        if hasattr(self, '_optimizer_step_count'):
                            print(f"  Total optimizer.step() calls: {self._optimizer_step_count}")
                    elif hasattr(self, '_optimizer_step_count'):
                        # Hook is running but no data stored yet (might be before step 20)
                        print(f"Optimizer.step() Hook: Active (calls: {self._optimizer_step_count}), data will appear at step 20+")
                    
                    print("="*70 + "\n")
                self._last_detailed_lr_log_step = current_step
        
        # Call parent log method to actually log everything (including our new learning rates)
        super().log(logs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()

        logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.config.save_pretrained(output_dir)

        keys_to_match = ['mm_projector', 'embed_tokens']
        weight_to_save = get_mm_projector_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
        torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))