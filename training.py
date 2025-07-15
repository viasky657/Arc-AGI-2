# --- ARC-AGI-2 Meta-Learning Training Loop ---
import os
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
import glob
import json
from dataclasses import dataclass, field
from typing import List, Optional, Any
import math

CUDA_LAUNCH_BLOCKING=1 #Diagnose cuda errors. 
# --- FIX: Define NUM_ARC_SYMBOLS globally for DataLoader workers ---
# The standard ARC task has 10 symbols (0-9).
NUM_ARC_SYMBOLS = 10
import numpy as np

print("\n" + "="*60)
print(f"🚀 STARTING PHASE 4: ARC-AGI-2 Meta-Learning Training")
print(f"   Epochs: {NUM_EPOCHS_ARC}, Batch Size: {ARC_BATCH_SIZE}, Task ID: {ARC_TASK_ID}")
print(f"   Device: {device if not accelerator_arc else accelerator_arc.device}")
print("="*60 + "\n")

# --- Principles Training Configuration ---
NUM_EPOCHS_PRINCIPLES = 3 #Can be lowered due to new DPPM++ Solver converging 10 epoch sooner.

# --- Training Configuration ---
USE_MIXED_PRECISION = torch.cuda.is_available()
autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
GRADIENT_ACCUMULATION_STEPS = 2
MAX_GRAD_NORM = 1.0
scaler = torch.amp.GradScaler('cuda',enabled=USE_MIXED_PRECISION)

# --- Context: 2D Grid Padding (from original code) ---
def pad_grid(grid_list, max_dims, pad_value):
    """Pads a 2D grid to specified maximum dimensions."""
    grid_np = np.array(grid_list, dtype=np.int32)
    padded_grid = np.full(max_dims, pad_value, dtype=np.int32)
    h, w = grid_np.shape
    padded_grid[:h, :w] = grid_np
    return padded_grid

# --- Fix: Byte Sequence Padding for the Model --- #
MAX_SEQUENCE_LENGTH = 8192
PADDING_BYTE_VALUE = 0

def serialize_and_pad_grid(grid, max_len=MAX_SEQUENCE_LENGTH, pad_value=PADDING_BYTE_VALUE):
    """
    Serializes a grid into a byte sequence and pads it to a fixed length.
    """
    flat_array = np.array(grid, dtype=np.uint8).flatten()
    byte_sequence = flat_array.tobytes()
    padding_len = max_len - len(byte_sequence)

    if padding_len < 0:
        padded_sequence = byte_sequence[:max_len]
    else:
        padding = bytes([pad_value] * padding_len)
        padded_sequence = byte_sequence + padding
        
    return padded_sequence

from contineous_thought_machines.models.ctm_Diffusion_NEWNEW import batched_numeric_tensor_to_bytes

class PrinciplesDataset(Dataset):
    def __init__(self, file_path, max_len=MAX_SEQUENCE_LENGTH, pad_value=PADDING_BYTE_VALUE, audio_duration_seconds=2.0, sample_rate=16000):
        self.max_len = max_len
        self.pad_value = pad_value
        self.audio_duration_seconds = audio_duration_seconds
        self.sample_rate = sample_rate
        self.principles = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.principles.append(line)
            print(f"PrinciplesDataset: Loaded {len(self.principles)} principles from {file_path}.")
        except Exception as e:
            print(f"PrinciplesDataset Warning: Could not load or parse {file_path}: {e}")

    def __len__(self):
        return len(self.principles)

    def __getitem__(self, idx):
        principle_text = self.principles[idx]
        
        # 1. Prepare text input
        text_bytes = torch.tensor(list(principle_text.encode('utf-8')), dtype=torch.uint8)
        
        # 2. Prepare silent audio template
        num_audio_samples = int(self.audio_duration_seconds * self.sample_rate)
        # Create on CPU, as conversion to bytes happens on CPU.
        audio_template_numeric = torch.zeros(1, num_audio_samples) # Batch of 1
        
        # Convert audio template to bytes
        audio_template_bytes = batched_numeric_tensor_to_bytes(audio_template_numeric, source_dtype=np.float32).squeeze(0)

        # 3. Create combined byte sequence
        separator = torch.tensor([255, 0, 255, 0, 255, 0, 255, 0], dtype=torch.uint8)
        
        combined_input_bytes = torch.cat([text_bytes, separator, audio_template_bytes])

        # 4. Pad or truncate the combined sequence
        padding_len = self.max_len - len(combined_input_bytes)
        if padding_len < 0:
            padded_sequence = combined_input_bytes[:self.max_len]
        else:
            padding = torch.full((padding_len,), self.pad_value, dtype=torch.uint8)
            padded_sequence = torch.cat([combined_input_bytes, padding])
            
        return padded_sequence

def collate_fn_principles(batch):
    # The batch is already a list of tensors from __getitem__
    # We just need to stack them.
    return {'input_byte_sequences': torch.stack(batch)}

class NewCustomARCGridDataset(Dataset):
    def __init__(self, data_dir, max_grid_size=MAX_GRID_SIZE, padding_value=PADDING_VALUE):
        self.data_dir = data_dir
        self.task_files = glob.glob(os.path.join(data_dir, "*.json"))
        self.max_grid_size = max_grid_size
        self.padding_value = padding_value
        self.tasks = []
        print(f"NewCustomARCGridDataset: Looking for tasks in: {data_dir}")
        if not self.task_files:
            print(f"NewCustomARCGridDataset Warning: No JSON files found in {data_dir}. Dataset will be empty.")
        for task_file in self.task_files:
            try:
                with open(task_file, 'r') as f:
                    self.tasks.append(json.load(f))
            except Exception as e:
                print(f"NewCustomARCGridDataset Warning: Could not load or parse {task_file}: {e}")
        if not self.tasks:
            print(f"NewCustomARCGridDataset Warning: No tasks successfully loaded from {data_dir}.")
        else:
            print(f"NewCustomARCGridDataset: Loaded {len(self.tasks)} ARC tasks from {data_dir}.")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task_data = self.tasks[idx]
        processed_task = {'train': [], 'test': [], 'id': os.path.basename(self.task_files[idx]) if idx < len(self.task_files) else 'unknown_task'}

        for pair_type in ['train', 'test']:
            for item in task_data.get(pair_type, []):
                input_grid_list = item.get('input', [])
                output_grid_list = item.get('output', [])
                
                original_input_dims = (len(input_grid_list), len(input_grid_list[0]) if input_grid_list and input_grid_list[0] else (0,0))
                original_output_dims = (len(output_grid_list), len(output_grid_list[0]) if output_grid_list and output_grid_list[0] else (0,0))

                padded_input_np = pad_grid(input_grid_list, self.max_grid_size, self.padding_value)
                padded_output_np = pad_grid(output_grid_list, self.max_grid_size, self.padding_value)
                
                processed_task[pair_type].append({
                    'input': torch.from_numpy(padded_input_np).long(),
                    'output': torch.from_numpy(padded_output_np).long(),
                    'original_input_dims': original_input_dims,
                    'original_output_dims': original_output_dims
                })
        return processed_task

def collate_fn_new_custom_arc(batch_of_tasks):
    input_byte_sequences_list = []
    target_byte_sequences_for_diffusion_list = []
    original_target_grids_for_ce_loss_list = []

    for task in batch_of_tasks:
        if not isinstance(task, dict):
            continue

        for train_pair in task.get('train', []):
            if not isinstance(train_pair, dict) or 'input' not in train_pair or 'output' not in train_pair:
                continue

            input_grid_np = train_pair['input'].numpy()
            target_grid_np = train_pair['output'].numpy()

            input_bytes = serialize_and_pad_grid(input_grid_np, max_len=MAX_SEQUENCE_LENGTH, pad_value=PADDING_BYTE_VALUE)
            input_byte_sequences_list.append(torch.tensor(list(input_bytes), dtype=torch.uint8))

            target_bytes_for_diffusion = serialize_and_pad_grid(target_grid_np, max_len=MAX_SEQUENCE_LENGTH, pad_value=PADDING_BYTE_VALUE)
            target_byte_sequences_for_diffusion_list.append(torch.tensor(list(target_bytes_for_diffusion), dtype=torch.uint8))

            original_target_grids_for_ce_loss_list.append(train_pair['output'].view(-1))
            
    if not input_byte_sequences_list:
        return {
            'input_byte_sequences': torch.empty(0, MAX_SEQUENCE_LENGTH, dtype=torch.uint8),
            'target_byte_sequences_for_diffusion': torch.empty(0, MAX_SEQUENCE_LENGTH, dtype=torch.uint8),
            'original_target_grids_for_ce_loss': torch.empty(0, ARC_INPUT_FLAT_DIM, dtype=torch.long),
        }

    final_input_byte_sequences = torch.stack(input_byte_sequences_list)
    final_target_byte_sequences_for_diffusion = torch.stack(target_byte_sequences_for_diffusion_list)
    final_original_target_grids_for_ce_loss = torch.stack(original_target_grids_for_ce_loss_list)
    
    # --- Fix for potential out-of-bounds padding values ---
    # The CrossEntropyLoss criterion expects class indices to be in [0, C-1].
    # If the padding value is negative or >= C, it can cause a CUDA 'device-side assert' error.
    # We defensively clamp the target tensor to the valid range [0, NUM_ARC_SYMBOLS - 1].
    final_original_target_grids_for_ce_loss.clamp_(min=0, max=NUM_ARC_SYMBOLS - 1)
    
    return {
        'input_byte_sequences': final_input_byte_sequences,
        'target_byte_sequences_for_diffusion': final_target_byte_sequences_for_diffusion,
        'original_target_grids_for_ce_loss': final_original_target_grids_for_ce_loss,
    }

arc_train_dataset = NewCustomARCGridDataset(ARC_TRAIN_DIR)
arc_eval_dataset = NewCustomARCGridDataset(ARC_EVAL_DIR)

arc_train_loader, arc_eval_loader = None, None
if arc_train_dataset and len(arc_train_dataset) > 0:
    arc_train_loader = DataLoader(
        arc_train_dataset, batch_size=ARC_BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn_new_custom_arc, **OPTIMIZED_DATALOADER_CONFIG
    )
    if accelerator_arc: arc_train_loader = accelerator_arc.prepare(arc_train_loader)
    print(f"✓ ARC Training DataLoader initialized with {len(arc_train_dataset)} tasks.")
else:
    print("⚠️ ARC Training DataLoader could not be initialized.")

if arc_eval_dataset and len(arc_eval_dataset) > 0:
    arc_eval_loader = DataLoader(
        arc_eval_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn_new_custom_arc, **OPTIMIZED_DATALOADER_CONFIG
    )
    if accelerator_arc: arc_eval_loader = accelerator_arc.prepare(arc_eval_loader)
    print(f"✓ ARC Evaluation DataLoader initialized with {len(arc_eval_dataset)} tasks.")
else:
    print("⚠️ ARC Evaluation DataLoader could not be initialized.")

'''
# --- Principles Dataset and DataLoader ---
PRINCIPLES_FILE_PATH = "contineous-thought-machines/models/Principles/principles.txt"
principles_dataset = PrinciplesDataset(PRINCIPLES_FILE_PATH)
principles_loader = None
if principles_dataset and len(principles_dataset) > 0:
    principles_loader = DataLoader(
        principles_dataset, batch_size=ARC_BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn_principles, **OPTIMIZED_DATALOADER_CONFIG
    )
    if accelerator_arc: principles_loader = accelerator_arc.prepare(principles_loader)
    print(f"✓ Principles DataLoader initialized with {len(principles_dataset)} principles.")
else:
    print("⚠️ Principles DataLoader could not be initialized.")
'''

# === DEBUG + RANK CHECK ===
def get_rank_debug():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    print(f"[DEBUG] Rank {rank} out of {world_size} total ranks")
    return rank, world_size


if not all([ctm_model_arc, optimizer_arc, arc_train_loader]):
    print("⚠️ Skipping ARC-AGI-2 training due to missing components.")
else:
    print("✓ All components ready for ARC training.")
    
    for epoch in range(NUM_EPOCHS_ARC):
        ctm_model_arc.train()
        if hasattr(ctm_model_arc, 'wake_up'):
            ctm_model_arc.wake_up()

        total_epoch_loss = 0
        
        progress_bar = tqdm(enumerate(arc_train_loader), total=len(arc_train_loader), desc=f"ARC Epoch {epoch + 1}")

        for batch_idx, batch_data in progress_bar:
            if not batch_data or batch_data['input_byte_sequences'].numel() == 0:
                print(f"Skipping empty batch {batch_idx}")
                continue

            input_bytes = batch_data['input_byte_sequences'].to(accelerator_arc.device if accelerator_arc else device)
            target_bytes_for_diffusion = batch_data['target_byte_sequences_for_diffusion'].to(accelerator_arc.device if accelerator_arc else device)
            
            current_batch_size = input_bytes.size(0)

            optimizer_arc.zero_grad()
            
            autocast_context = accelerator_arc.autocast() if accelerator_arc else autocast(enabled=USE_MIXED_PRECISION, dtype=autocast_dtype)

            with autocast_context:
                model_output_dict = ctm_model_arc(
                    byte_sequence=input_bytes,
                    target_diffusion_output=target_bytes_for_diffusion,
                    mode='ctm_controlled_diffusion',
                    timestep=torch.randint(0, config_arc_diffusion.diffusion_steps, (current_batch_size,), device=input_bytes.device).long()
                )

                total_loss = model_output_dict.get('total_loss', torch.tensor(0.0, device=input_bytes.device))

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"[NaN or Inf Loss Detected] at Epoch {epoch+1}, Batch {batch_idx+1}. Skipping backward pass.")
                continue

            if accelerator_arc:
                accelerator_arc.backward(total_loss)
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                   if accelerator_arc.sync_gradients:
                       accelerator_arc.clip_grad_norm_(ctm_model_arc.parameters(), MAX_GRAD_NORM)
                   optimizer_arc.step()
                   optimizer_arc.zero_grad()
            else:
                scaler.scale(total_loss).backward()
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer_arc)
                    torch.nn.utils.clip_grad_norm_(ctm_model_arc.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer_arc)
                    scaler.update()
                    optimizer_arc.zero_grad()

            total_epoch_loss += total_loss.item()
            progress_bar.set_postfix({
                'loss': total_loss.item(),
                'avg_loss': total_epoch_loss / (batch_idx + 1)
            })

        avg_epoch_loss = total_epoch_loss / len(arc_train_loader) if len(arc_train_loader) > 0 else 0
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_ARC}] completed. Average Loss: {avg_epoch_loss:.4f}")

        # --- Evaluation Step ---
        ctm_model_arc.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for eval_batch_data in arc_eval_loader:
                input_bytes = eval_batch_data['input_byte_sequences'].to(accelerator_arc.device if accelerator_arc else device)
                target_bytes = eval_batch_data['target_byte_sequences_for_diffusion'].to(accelerator_arc.device if accelerator_arc else device)
                
                with autocast_context:
                    eval_output = ctm_model_arc(
                        byte_sequence=input_bytes,
                        target_diffusion_output=target_bytes,
                        mode='ctm_controlled_diffusion',
                        timestep=torch.randint(0, config_arc_diffusion.diffusion_steps, (input_bytes.size(0),), device=input_bytes.device).long()
                    )
                    eval_loss = eval_output.get('total_loss', torch.tensor(0.0, device=input_bytes.device))
                total_eval_loss += eval_loss.item()
        
        avg_eval_loss = total_eval_loss / len(arc_eval_loader) if len(arc_eval_loader) > 0 else 0
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_ARC}] Evaluation Loss: {avg_eval_loss:.4f}")

        # --- Checkpointing ---
        if accelerator_arc and accelerator_arc.is_main_process:
            if (epoch + 1) % 5 == 0: # Save every 5 epochs
                accelerator_arc.wait_for_everyone()
                unwrapped_model = accelerator_arc.unwrap_model(ctm_model_arc)
                
                # --- DeepSpeed Check ---
                if hasattr(accelerator_arc.state, 'deepspeed_plugin') and accelerator_arc.state.deepspeed_plugin is not None:
                    # DeepSpeed handles checkpointing via accelerator.save_state
                    accelerator_arc.save_state(os.path.join(CHECKPOINT_DIR_ARC, f"epoch_{epoch+1}"))
                else:
                    # For other setups, save with safetensors on rank 0
                    save_file(unwrapped_model.state_dict(), os.path.join(CHECKPOINT_DIR_ARC, f"ctm_model_arc_epoch_{epoch+1}.safetensors"))
                
                print(f"✓ Checkpoint saved for epoch {epoch+1} to {CHECKPOINT_DIR_ARC}")

        if hasattr(ctm_model_arc, 'sleep_down'):
            ctm_model_arc.sleep_down()

    print("\n🎉 ARC-AGI-2 Meta-Learning Training Phase Completed!")

'''
# --- Principles Alignment Training Loop ---
if principles_loader and NUM_EPOCHS_PRINCIPLES > 0 and 'ctm_model_arc' in globals() and ctm_model_arc is not None:
    print("\n" + "="*60)
    print(f"🚀 STARTING PHASE: Principles Alignment Training")
    print(f"   Epochs: {NUM_EPOCHS_PRINCIPLES}")
    print("="*60 + "\n")

    for epoch in range(NUM_EPOCHS_PRINCIPLES):
        ctm_model_arc.train()
        if hasattr(ctm_model_arc, 'wake_up'):
            ctm_model_arc.wake_up()

        total_epoch_loss_principles = 0
        
        progress_bar_principles = tqdm(enumerate(principles_loader), total=len(principles_loader), desc=f"Principles Epoch {epoch + 1}")

        for batch_idx, batch_data in progress_bar_principles:
            if not batch_data or batch_data['input_byte_sequences'].numel() == 0:
                print(f"Skipping empty principles batch {batch_idx}")
                continue

            input_bytes = batch_data['input_byte_sequences'].to(accelerator_arc.device if accelerator_arc else device)
            
            current_batch_size = input_bytes.size(0)

            optimizer_arc.zero_grad()
            
            autocast_context = accelerator_arc.autocast() if accelerator_arc else autocast(enabled=USE_MIXED_PRECISION, dtype=autocast_dtype)

            with autocast_context:
                # For principles, the input is the target. The model learns to reconstruct the principles.
                model_output_dict = ctm_model_arc(
                    byte_sequence=input_bytes,
                    target_diffusion_output=input_bytes, # Self-supervision
                    mode='ctm_controlled_diffusion',
                    timestep=torch.randint(0, config_arc_diffusion.diffusion_steps, (current_batch_size,), device=input_bytes.device).long()
                )

                total_loss = model_output_dict.get('total_loss', torch.tensor(0.0, device=input_bytes.device))

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"[NaN or Inf Loss Detected] in Principles training at Epoch {epoch+1}, Batch {batch_idx+1}. Skipping backward pass.")
                continue

            if accelerator_arc:
                accelerator_arc.backward(total_loss)
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                   if accelerator_arc.sync_gradients:
                       accelerator_arc.clip_grad_norm_(ctm_model_arc.parameters(), MAX_GRAD_NORM)
                   optimizer_arc.step()
                   optimizer_arc.zero_grad()
            else:
                scaler.scale(total_loss).backward()
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer_arc)
                    torch.nn.utils.clip_grad_norm_(ctm_model_arc.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer_arc)
                    scaler.update()
                    optimizer_arc.zero_grad()

            total_epoch_loss_principles += total_loss.item()
            progress_bar_principles.set_postfix({
                'loss': total_loss.item(),
                'avg_loss': total_epoch_loss_principles / (batch_idx + 1)
            })

        avg_epoch_loss_principles = total_epoch_loss_principles / len(principles_loader) if len(principles_loader) > 0 else 0
        print(f"Principles Epoch [{epoch+1}/{NUM_EPOCHS_PRINCIPLES}] completed. Average Loss: {avg_epoch_loss_principles:.4f}")

        # --- Checkpointing for Principles Training ---
        if accelerator_arc and accelerator_arc.is_main_process:
            if (epoch + 1) % 5 == 0: # Save every 5 epochs
                accelerator_arc.wait_for_everyone()
                unwrapped_model = accelerator_arc.unwrap_model(ctm_model_arc)
                
                checkpoint_dir = os.path.join(CHECKPOINT_DIR_ARC, "principles_checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)

                if hasattr(accelerator_arc.state, 'deepspeed_plugin') and accelerator_arc.state.deepspeed_plugin is not None:
                    accelerator_arc.save_state(os.path.join(checkpoint_dir, f"epoch_{epoch+1}"))
                else:
                    save_file(unwrapped_model.state_dict(), os.path.join(checkpoint_dir, f"ctm_model_arc_epoch_{epoch+1}.safetensors"))
                
                print(f"✓ Principles checkpoint saved for epoch {epoch+1} to {checkpoint_dir}")

        if hasattr(ctm_model_arc, 'sleep_down'):
            ctm_model_arc.sleep_down()

    print("\n🎉 Principles Alignment Training Phase Completed!")
'''
#The Mixed Context training is not needed since the Program Synthesizer is not being used and the CTM Nueron Network is being used instead. 
# --- Mixed Context Training ---
'''
import random

class MixedContextDataset(Dataset):
    def __init__(self, num_samples=1000, short_len=256, long_len=4096, vocab_size=256):
        self.num_samples = num_samples
        self.short_len = short_len
        self.long_len = long_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if random.random() < 0.5:
            # Short sequence, pack multiple
            num_packed = self.long_len // self.short_len
            packed = []
            mask = torch.zeros(self.long_len, self.long_len)
            pos = 0
            for i in range(num_packed):
                seq = torch.randint(0, self.vocab_size, (self.short_len,))
                packed.append(seq)
                # Causal mask for this segment
                segment_mask = torch.tril(torch.ones(self.short_len, self.short_len))
                mask[pos:pos+self.short_len, pos:pos+self.short_len] = segment_mask
                pos += self.short_len
            sequence = torch.cat(packed)[:self.long_len]
            is_long = False
        else:
            # Long sequence
            sequence = torch.randint(0, self.vocab_size, (self.long_len,))
            mask = torch.tril(torch.ones(self.long_len, self.long_len))
            is_long = True

        return {'sequence': sequence, 'mask': mask, 'is_long': is_long}

#mixed_dataset = MixedContextDataset()

#mixed_loader = DataLoader(mixed_dataset, batch_size=4, shuffle=True)

# Mixed training loop
 for epoch in range(5): 
     ctm_model_arc.train()
     total_loss = 0
     for batch in mixed_loader:
        sequence = batch['sequence'].to(device)
        attn_mask = batch['mask'].to(device)
        is_long = batch['is_long']

         Assuming model has train_forward that computes loss
        loss = ctm_model_arc.train_forward(sequence, attn_mask, use_rescaled_rope=is_long)

        optimizer_arc.zero_grad()
        loss.backward()
        optimizer_arc.step()
        total_loss += loss.item()

    print(f"Mixed Context Epoch {epoch+1} Avg Loss: {total_loss / len(mixed_loader)}")

print("\n🎉 Mixed Context Training Completed!")
'''