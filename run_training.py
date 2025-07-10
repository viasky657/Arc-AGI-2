import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
import glob
import json
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Tuple, Union, Callable
import math
import numpy as np
import warnings
from accelerate import Accelerator
from contineous_thought_machines.models.ctm_Diffusion_NEWNEW import batched_numeric_tensor_to_bytes

# --- Path Setup ---
project_root = '/workspaces/Arc-AGI-2'
paths_to_add = [
    os.path.join(project_root, 'contineous-thought-machines'),
    os.path.join(project_root, 'contineous-thought-machines', 'models'),
]
for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# --- Model Import ---
try:
    from contineous_thought_machines.models.ctm_Diffusion_NEWNEW import EnhancedCTMDiffusion
    print("✓ Successfully imported EnhancedCTMDiffusion model.")
except ImportError as e:
    print(f"Error importing Enhanced CTM: {e}")
    EnhancedCTMDiffusion = None

# --- Main Configuration ---
ARC_TRAIN_DIR = "/workspaces/Arc-AGI-2/contineous-thought-machines/data/training"
ARC_EVAL_DIR = "/workspaces/Arc-AGI-2/contineous-thought-machines/data/evaluation"
CHECKPOINT_DIR = "checkpoints"
PRINCIPLES_FILE_PATH = "contineous_thought_machines/models/Principles/principles.txt"

MAX_GRID_SIZE = (30, 30)
NUM_ARC_SYMBOLS = 10
PADDING_VALUE = -1
ARC_INPUT_FLAT_DIM = MAX_GRID_SIZE[0] * MAX_GRID_SIZE[1]
MAX_SEQUENCE_LENGTH = 8192
PADDING_BYTE_VALUE = 0

LEARNING_RATE = 1e-4
NUM_EPOCHS_ARC = 20
NUM_EPOCHS_PRINCIPLES = 10
ARC_BATCH_SIZE = 16  # Increased for better parallelism
GRADIENT_ACCUMULATION_STEPS = 2 # Decreased to maintain effective batch size
MAX_GRAD_NORM = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"
USE_MIXED_PRECISION = torch.cuda.is_available()
autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.amp.GradScaler('cuda', enabled=USE_MIXED_PRECISION)

OPTIMIZED_DATALOADER_CONFIG = {
    "num_workers": min(os.cpu_count(), 4),
    "pin_memory": True,
    "prefetch_factor": 2
} if torch.cuda.is_available() else {}


# --- Enhanced CTM Configuration Definition ---
@dataclass
class EnhancedCTMConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 24
    max_sequence_length: int = 8192
    dropout: float = 0.1
    patch_embedding_dim: int = 256
    patch_encoder_cnn_channels: int = 64
    use_dynamic_entropy_patcher: bool = True
    entropy_patcher_threshold_type: str = "global"
    entropy_patcher_global_threshold: float = 0.75
    entropy_patcher_min_patch_size: int = 4
    entropy_patcher_max_patch_size: int = 128
    ctm_iterations: int = 5
    ctm_d_model: int = 512
    ctm_input_dim: int = 256
    ctm_heads: int = 8
    ctm_out_dims: int = 512
    ctm_neuron_select_type: str = 'bio_multi_objective'
    diffusion_steps: int = 1000
    attention_type: str = "subquadratic"
    positional_embedding_type: Optional[str] = 'multi-learnable-fourier'
    reshape_patch_sequence_to_grid: bool = True
    patch_grid_width: Optional[int] = None
    enable_pipeline_parallelism: bool = True
    pipeline_stages: int = 4
    enable_adaptive_batching: bool = True
    initial_batch_size: int = 32
    min_batch_size: int = 8
    max_batch_size: int = 256
    vocab_size: Optional[int] = None
    output_audio_bytes: bool = True
    unet_input_feature_dim: Optional[int] = None
    enable_consciousness_controller: bool = True
    consciousness_max_attention_steps: int = 100
    # Add other fields from Setup.py's config here if needed

    def __post_init__(self):
        if self.reshape_patch_sequence_to_grid and (self.patch_grid_width is None or self.patch_grid_width <= 0):
            raise ValueError("patch_grid_width must be positive if reshaping.")
        if self.unet_input_feature_dim is None:
            self.unet_input_feature_dim = self.max_sequence_length // 4

# --- Helper Functions and Datasets ---

def pad_grid(grid_list, max_dims, pad_value):
    grid_np = np.array(grid_list, dtype=np.int32)
    padded_grid = np.full(max_dims, pad_value, dtype=np.int32)
    h, w = grid_np.shape
    padded_grid[:h, :w] = grid_np
    return padded_grid

def serialize_and_pad_grid(grid, max_len=MAX_SEQUENCE_LENGTH, pad_value=PADDING_BYTE_VALUE):
    flat_array = np.array(grid, dtype=np.uint8).flatten()
    byte_sequence = flat_array.tobytes()
    padding_len = max_len - len(byte_sequence)
    if padding_len < 0:
        padded_sequence = byte_sequence[:max_len]
    else:
        padding = bytes([pad_value] * padding_len)
        padded_sequence = byte_sequence + padding
    return padded_sequence

class PrinciplesDataset(Dataset):
    def __init__(self, file_path, max_len=MAX_SEQUENCE_LENGTH, pad_value=PADDING_BYTE_VALUE, audio_duration_seconds=2.0, sample_rate=16000):
        self.max_len = max_len
        self.pad_value = pad_value
        self.audio_duration_seconds = audio_duration_seconds
        self.sample_rate = sample_rate
        self.principles = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.principles = [line.strip() for line in f if line.strip()]
            print(f"PrinciplesDataset: Loaded {len(self.principles)} principles.")
        except Exception as e:
            print(f"PrinciplesDataset Warning: Could not load {file_path}: {e}")

    def __len__(self):
        return len(self.principles)

    def __getitem__(self, idx):
        text_bytes = torch.tensor(list(self.principles[idx].encode('utf-8')), dtype=torch.uint8)
        num_audio_samples = int(self.audio_duration_seconds * self.sample_rate)
        audio_template_numeric = torch.zeros(1, num_audio_samples)
        audio_template_bytes = batched_numeric_tensor_to_bytes(audio_template_numeric, source_dtype=np.float32).squeeze(0)
        separator = torch.tensor([255, 0, 255, 0, 255, 0, 255, 0], dtype=torch.uint8)
        combined_input_bytes = torch.cat([text_bytes, separator, audio_template_bytes])
        padding_len = self.max_len - len(combined_input_bytes)
        if padding_len < 0:
            return combined_input_bytes[:self.max_len]
        padding = torch.full((padding_len,), self.pad_value, dtype=torch.uint8)
        return torch.cat([combined_input_bytes, padding])

def collate_fn_principles(batch):
    return {'input_byte_sequences': torch.stack(batch)}

class NewCustomARCGridDataset(Dataset):
    def __init__(self, data_dir, max_grid_size=MAX_GRID_SIZE, padding_value=PADDING_VALUE):
        self.task_files = glob.glob(os.path.join(data_dir, "*.json"))
        self.max_grid_size = max_grid_size
        self.padding_value = padding_value
        self.tasks = []
        print(f"NewCustomARCGridDataset: Looking for tasks in: {data_dir}")
        if not self.task_files:
            print(f"Warning: No JSON files found in {data_dir}.")
        for task_file in self.task_files:
            try:
                with open(task_file, 'r') as f:
                    self.tasks.append(json.load(f))
            except Exception as e:
                print(f"Warning: Could not load or parse {task_file}: {e}")
        print(f"NewCustomARCGridDataset: Loaded {len(self.tasks)} ARC tasks.")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task_data = self.tasks[idx]
        processed_task = {'train': [], 'test': [], 'id': os.path.basename(self.task_files[idx])}
        for pair_type in ['train', 'test']:
            for item in task_data.get(pair_type, []):
                input_grid_list = item.get('input', [])
                output_grid_list = item.get('output', [])
                padded_input_np = pad_grid(input_grid_list, self.max_grid_size, self.padding_value)
                padded_output_np = pad_grid(output_grid_list, self.max_grid_size, self.padding_value)
                processed_task[pair_type].append({
                    'input': torch.from_numpy(padded_input_np).long(),
                    'output': torch.from_numpy(padded_output_np).long()
                })
        return processed_task

def collate_fn_new_custom_arc(batch_of_tasks):
    input_bytes_list, target_bytes_list, target_grids_list = [], [], []
    for task in batch_of_tasks:
        for train_pair in task.get('train', []):
            input_bytes = serialize_and_pad_grid(train_pair['input'].numpy())
            input_bytes_list.append(torch.tensor(list(input_bytes), dtype=torch.uint8))
            target_bytes = serialize_and_pad_grid(train_pair['output'].numpy())
            target_bytes_list.append(torch.tensor(list(target_bytes), dtype=torch.uint8))
            target_grids_list.append(train_pair['output'].view(-1))
    if not input_bytes_list:
        return {}
    final_target_grids = torch.stack(target_grids_list)
    final_target_grids.clamp_(min=0, max=NUM_ARC_SYMBOLS - 1)
    return {
        'input_byte_sequences': torch.stack(input_bytes_list),
        'target_byte_sequences_for_diffusion': torch.stack(target_bytes_list),
        'original_target_grids_for_ce_loss': final_target_grids,
    }

# --- Main Script ---
if __name__ == "__main__":
    print("--- Initializing Model and Optimizer ---")
    config_arc_diffusion = EnhancedCTMConfig(patch_grid_width=16, unet_input_feature_dim=MAX_SEQUENCE_LENGTH // 4)
    
    if EnhancedCTMDiffusion is None:
        sys.exit("Model import failed. Exiting.")

    ctm_model_arc = EnhancedCTMDiffusion(config=config_arc_diffusion)
    optimizer_arc = optim.AdamW(ctm_model_arc.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    print("--- Initializing Accelerator ---")
    accelerator_arc = Accelerator()
    device = accelerator_arc.device
    ctm_model_arc.to(device)
    ctm_model_arc, optimizer_arc = accelerator_arc.prepare(ctm_model_arc, optimizer_arc)
    
    print("--- Initializing Dataloaders ---")
    arc_train_dataset = NewCustomARCGridDataset(ARC_TRAIN_DIR)
    arc_eval_dataset = NewCustomARCGridDataset(ARC_EVAL_DIR)
    principles_dataset = PrinciplesDataset(PRINCIPLES_FILE_PATH)

    arc_train_loader = DataLoader(
        arc_train_dataset, batch_size=ARC_BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn_new_custom_arc, **OPTIMIZED_DATALOADER_CONFIG
    )
    arc_eval_loader = DataLoader(
        arc_eval_dataset, batch_size=1, shuffle=False, # Eval batch size is 1
        collate_fn=collate_fn_new_custom_arc, **OPTIMIZED_DATALOADER_CONFIG
    )
    principles_loader = DataLoader(
        principles_dataset, batch_size=ARC_BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn_principles, **OPTIMIZED_DATALOADER_CONFIG
    )

    arc_train_loader, arc_eval_loader, principles_loader = accelerator_arc.prepare(
        arc_train_loader, arc_eval_loader, principles_loader
    )

    CHECKPOINT_DIR_ARC = os.path.join(CHECKPOINT_DIR, "ctm_arc_agi_2_enhanced_diffusion")
    os.makedirs(CHECKPOINT_DIR_ARC, exist_ok=True)
    print(f"Checkpoints will be saved to: {CHECKPOINT_DIR_ARC}")

    # --- ARC Training Loop ---
    print("\n" + "="*60 + "\n🚀 STARTING ARC-AGI-2 Meta-Learning Training\n" + "="*60)
    for epoch in range(NUM_EPOCHS_ARC):
        ctm_model_arc.train()
        total_epoch_loss = 0
        progress_bar = tqdm(enumerate(arc_train_loader), total=len(arc_train_loader), desc=f"ARC Epoch {epoch + 1}")

        for batch_idx, batch_data in progress_bar:
            if not batch_data or batch_data['input_byte_sequences'].numel() == 0:
                continue

            input_bytes = batch_data['input_byte_sequences']
            target_bytes = batch_data['target_byte_sequences_for_diffusion']
            
            with accelerator_arc.accumulate(ctm_model_arc):
                optimizer_arc.zero_grad()
                with autocast(enabled=USE_MIXED_PRECISION):
                    model_output = ctm_model_arc(
                        byte_sequence=input_bytes,
                        target_diffusion_output=target_bytes,
                        mode='ctm_controlled_diffusion',
                        timestep=torch.randint(0, config_arc_diffusion.diffusion_steps, (input_bytes.size(0),), device=device).long()
                    )
                    loss = model_output.get('total_loss')

                if loss is not None and torch.isfinite(loss):
                    accelerator_arc.backward(loss)
                    if accelerator_arc.sync_gradients:
                        accelerator_arc.clip_grad_norm_(ctm_model_arc.parameters(), MAX_GRAD_NORM)
                    optimizer_arc.step()
                    total_epoch_loss += loss.item()
                    progress_bar.set_postfix({'loss': loss.item()})

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_ARC}] completed. Avg Loss: {total_epoch_loss / len(arc_train_loader):.4f}")

        # --- Evaluation Step ---
        # ... (Evaluation logic can be added here if desired) ...

        # --- Checkpointing ---
        if accelerator_arc.is_main_process and (epoch + 1) % 5 == 0:
            accelerator_arc.wait_for_everyone()
            unwrapped_model = accelerator_arc.unwrap_model(ctm_model_arc)
            save_path = os.path.join(CHECKPOINT_DIR_ARC, f"ctm_model_arc_epoch_{epoch+1}.safetensors")
            save_file(unwrapped_model.state_dict(), save_path)
            print(f"✓ Checkpoint saved to {save_path}")

    print("\n🎉 ARC-AGI-2 Meta-Learning Training Phase Completed!")

    # --- Principles Alignment Training Loop (Disabled) ---
    # print("\n" + "="*60 + "\n🚀 STARTING Principles Alignment Training\n" + "="*60)
    # for epoch in range(NUM_EPOCHS_PRINCIPLES):
    #     ctm_model_arc.train()
    #     total_epoch_loss = 0
    #     progress_bar = tqdm(principles_loader, desc=f"Principles Epoch {epoch + 1}")
    #     for batch_data in progress_bar:
    #          if not batch_data or batch_data['input_byte_sequences'].numel() == 0:
    #             continue
    #          input_bytes = batch_data['input_byte_sequences']
    #          with accelerator_arc.accumulate(ctm_model_arc):
    #             optimizer_arc.zero_grad()
    #             with autocast(enabled=USE_MIXED_PRECISION):
    #                 model_output = ctm_model_arc(
    #                     byte_sequence=input_bytes,
    #                     target_diffusion_output=input_bytes, # Self-supervision
    #                     mode='ctm_controlled_diffusion',
    #                     timestep=torch.randint(0, config_arc_diffusion.diffusion_steps, (input_bytes.size(0),), device=device).long()
    #                 )
    #                 loss = model_output.get('total_loss')
    #             if loss is not None and torch.isfinite(loss):
    #                 accelerator_arc.backward(loss)
    #                 if accelerator_arc.sync_gradients:
    #                      accelerator_arc.clip_grad_norm_(ctm_model_arc.parameters(), MAX_GRAD_NORM)
    #                 optimizer_arc.step()
    #                 total_epoch_loss += loss.item()
    #                 progress_bar.set_postfix({'loss': loss.item()})
    #     print(f"Principles Epoch [{epoch+1}/{NUM_EPOCHS_PRINCIPLES}] completed. Avg Loss: {total_epoch_loss / len(principles_loader):.4f}")
    
    # print("\n🎉 Principles Alignment Training Phase Completed!")
