# --- ARC-AGI-2 Meta-Learning Training Loop ---
import os
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
import glob
import json

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

# --- MCMC Plasticity Loss Normalization Factor ---
# Scale MCMC loss to prevent overpowering other loss components
MCMC_LOSS_SCALE = 0.01  # Scale factor for MCMC loss
MAX_MCMC_LOSS_FOR_PLASTICITY = 1000  # Increased cap to allow larger raw loss values

if not all([ctm_model_arc, arc_output_head, optimizer_arc, arc_train_loader, arc_criterion]):
    print("⚠️ Skipping ARC-AGI-2 training due to missing components.")
else:
    # Record original global plasticity weight for scheduling
    orig_global_plasticity_loss_weight = ctm_model_arc.global_plasticity_loss_weight
    orig_local_selector_loss_weight = ctm_model_arc.local_neuron_selector_loss_weight
    for epoch in range(NUM_EPOCHS_ARC):

        if epoch < 10: # Linear ramp-up of global plasticity weight over first 10 epochs
            ramp_factor = (epoch + 1) / 10.0
            ctm_model_arc.global_plasticity_loss_weight = orig_global_plasticity_loss_weight * ramp_factor
        else: # Full weight training for the global plasticity weight
            ctm_model_arc.global_plasticity_loss_weight = orig_global_plasticity_loss_weight
            ctm_model_arc.train()
            arc_output_head.train()
            if ctm_mcmc_integration_arc: ctm_mcmc_integration_arc.train()

            total_arc_loss = 0
            processed_batches = 0

        total_arc_loss = 0
        processed_batches = 0
        for batch_idx, batch_data in enumerate(arc_train_loader):
            if not batch_data or batch_data['input_byte_sequences'].numel() == 0:
                print(f"Skipping empty batch {batch_idx}")
                continue

            # Get data from the updated collate_fn
            input_bytes = batch_data['input_byte_sequences'].to(device if not accelerator_arc else accelerator_arc.device)
            target_bytes_for_diffusion = batch_data['target_byte_sequences_for_diffusion'].to(device if not accelerator_arc else accelerator_arc.device)
            original_target_grids_for_ce = batch_data['original_target_grids_for_ce_loss'].to(device if not accelerator_arc else accelerator_arc.device)

            current_batch_size = input_bytes.size(0)

            optimizer_arc.zero_grad()

            with autocast(enabled=USE_MIXED_PRECISION, dtype=autocast_dtype) if not accelerator_arc else accelerator_arc.autocast():
                # Forward pass through EnhancedCTMDiffusion
                model_output_dict = ctm_model_arc(
                    byte_sequence=input_bytes,
                    target_diffusion_output=target_bytes_for_diffusion,
                    mode='ctm_controlled_diffusion',
                    timestep=torch.randint(0, config_arc_diffusion.diffusion_steps, (current_batch_size,), device=input_bytes.device).long(),
                    target_mcmc_output=None,
                    task_name="ARC_AGI_2",
                    current_epoch=epoch
                )

                # Retrieve CTM output components including all model-internal losses and signals
                diffusion_loss = model_output_dict.get('diffusion_loss', torch.tensor(0.0, device=input_bytes.device))
                predictions = model_output_dict.get('predictions', None)
                certainties = model_output_dict.get('certainties', None)
                final_sync_out = model_output_dict.get('final_sync_out', None)
                predictive_coding_loss = model_output_dict.get('predictive_coding_loss', torch.tensor(0.0, device=input_bytes.device))
                local_hebbian_signal = model_output_dict.get('local_hebbian_signal', torch.tensor(0.0, device=input_bytes.device))
                local_neuron_selector_loss_model = model_output_dict.get('local_neuron_selector_loss', torch.tensor(0.0, device=input_bytes.device))
                global_plasticity_loss = model_output_dict.get('global_plasticity_loss', torch.tensor(0.0, device=input_bytes.device))
                # Initialize total loss for the optimizer with the model's computed total_loss
                total_loss = model_output_dict.get('total_loss', diffusion_loss)
                
                # --- Get CTM core output for auxiliary heads ---
                ctm_backbone_output = None
                if model_output_dict and 'predictions' in model_output_dict:
                    ctm_backbone_output = model_output_dict['predictions'][:, :, -1]
                elif model_output_dict and 'final_sync_out' in model_output_dict:
                    ctm_backbone_output = model_output_dict['final_sync_out']
                else:
                    print("Warning: CTM core output not found. Using zeros for auxiliary head inputs.")
                    ctm_backbone_output = torch.zeros(current_batch_size, config_arc_diffusion.ctm_out_dims, device=input_bytes.device)
                
                # --- Calculate and add auxiliary losses to the total_loss for the optimizer ---
                ce_loss = torch.tensor(0.0, device=input_bytes.device)
                if arc_output_head and ctm_backbone_output is not None:
                    if ctm_backbone_output.ndim > 2:
                        ctm_features_for_head = ctm_backbone_output.mean(dim=1)
                    else:
                        ctm_features_for_head = ctm_backbone_output
                    
                    predicted_logits = arc_output_head(torch.tanh(ctm_features_for_head))
                    predicted_logits_reshaped = predicted_logits.view(current_batch_size * ARC_INPUT_FLAT_DIM, NUM_ARC_SYMBOLS)
                    target_grids_reshaped = original_target_grids_for_ce.view(current_batch_size * ARC_INPUT_FLAT_DIM)
                    ce_loss = arc_criterion(predicted_logits_reshaped, target_grids_reshaped)
                    total_loss = total_loss + ce_loss

                mcmc_loss_val = torch.tensor(0.0, device=input_bytes.device)
                norm_mcmc_loss_for_plasticity = torch.tensor(0.0, device=input_bytes.device)
                if ctm_mcmc_integration_arc and ctm_backbone_output is not None:
                    target_grids_for_mcmc = (original_target_grids_for_ce > 0).float()
                    # Apply y-normalization for MCMC target
                    y_mean = target_grids_for_mcmc.mean()
                    y_std = target_grids_for_mcmc.std()
                    normalized_target_y = (target_grids_for_mcmc - y_mean) / (y_std + 1e-8)

                    mcmc_input_features = ctm_backbone_output.detach()
                    if mcmc_input_features.ndim > 2:
                        mcmc_input_features = mcmc_input_features.mean(dim=1)

                    mcmc_loss_val, _, _ = ctm_mcmc_integration_arc(x=mcmc_input_features, target_y=normalized_target_y)
                    
                    # --- STABILITY FIX: Clamp MCMC loss to prevent explosion ---
                    # Apply scaling and clamping to MCMC loss
                    mcmc_loss_val = mcmc_loss_val * MCMC_LOSS_SCALE
                    mcmc_loss_val = torch.clamp(mcmc_loss_val, -MAX_MCMC_LOSS_FOR_PLASTICITY, MAX_MCMC_LOSS_FOR_PLASTICITY)
                    total_loss = total_loss + mcmc_loss_val
                    
                    # --- STABILITY FIX: MCMC loss normalization for plasticity ---
                    # Use the scaled loss for plasticity calculations
                    mcmc_for_plasticity = mcmc_loss_val.detach()
                    # Apply tanh to bound the plasticity signal between -1 and 1
                    norm_mcmc_loss_for_plasticity = torch.tanh(mcmc_for_plasticity)

                    # --- Dynamic scaling with bounds ---
                    abs_hebbian_mean = local_hebbian_signal.abs().mean()
                    # Clamp the denominator to prevent division by very small numbers
                    abs_hebbian_mean_clamped = torch.clamp(abs_hebbian_mean, min=1e-4, max=10.0)
                    dyn_lambda = torch.clamp(orig_local_selector_loss_weight / abs_hebbian_mean_clamped, min=0.01, max=10.0)
                    dynamic_hebbian_loss = dyn_lambda * abs_hebbian_mean_clamped
                    total_loss = total_loss + dynamic_hebbian_loss

            # --- Enhanced NaN/Inf Check and Loss Debugging ---
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"[NaN or Inf Loss Detected] at Epoch {epoch+1}, Batch {batch_idx+1}. Skipping backward pass.")
                print(f"  - Diffusion Loss: {diffusion_loss.item() if torch.isfinite(diffusion_loss) else 'NaN/Inf'}")
                print(f"  - CE Loss: {ce_loss.item() if torch.isfinite(ce_loss) else 'NaN/Inf'}")
                print(f"  - MCMC Loss: {mcmc_loss_val.item() if torch.isfinite(mcmc_loss_val) else 'NaN/Inf'}")
                print(f"  - Dynamic Hebbian Loss: {dynamic_hebbian_loss.item() if torch.isfinite(dynamic_hebbian_loss) else 'NaN/Inf'}")
                continue # Skip to the next batch

            # --- Enhanced Loss Monitoring ---
            if (batch_idx + 1) % 10 == 0:  # Print every 10 batches instead of every batch
                print(f"[Losses] Diff: {diffusion_loss.item():.4f}, CE: {ce_loss.item():.4f}, MCMC: {mcmc_loss_val.item():.4f}, Dyn_Heb: {dynamic_hebbian_loss.item():.4f}, Total: {total_loss.item():.4f}")
                
                # --- MCMC Loss Monitoring ---
                if abs(mcmc_loss_val.item()) > 5.0:  # Alert if MCMC loss is getting large
                    print(f"[WARNING] Large MCMC loss detected: {mcmc_loss_val.item():.4f}")

            if scaler:
                scaler.scale(total_loss).backward()
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer_arc)
                    
                    # --- Enhanced Gradient Clipping ---
                    total_norm = torch.nn.utils.clip_grad_norm_(ctm_model_arc.parameters(), MAX_GRAD_NORM)
                    if total_norm > MAX_GRAD_NORM * 2:  # Alert if gradients are very large
                        print(f"[WARNING] Large gradient norm detected: {total_norm:.4f}")
                    
                    # --- Activity-Dependent Plasticity with Enhanced Stability ---
                    unwrapped_model = ctm_model_arc
                    # Enhanced plasticity call with clamped losses
                    clamped_diffusion_loss = torch.clamp(diffusion_loss, -10.0, 10.0)
                    clamped_ce_loss = torch.clamp(ce_loss, -10.0, 10.0)
                    unwrapped_model.ctm_core.apply_activity_plasticity(clamped_diffusion_loss, clamped_ce_loss, norm_mcmc_loss_for_plasticity)
                    scaler.step(optimizer_arc)
                    scaler.update()
                    optimizer_arc.zero_grad(set_to_none=True)
            elif accelerator_arc:
                accelerator_arc.backward(total_loss)
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                   # --- Enhanced Gradient Clipping for Accelerator ---
                   total_norm = torch.nn.utils.clip_grad_norm_(ctm_model_arc.parameters(), MAX_GRAD_NORM)
                   if total_norm > MAX_GRAD_NORM * 2:
                       print(f"[WARNING] Large gradient norm detected (accelerator): {total_norm:.4f}")
                   
                   # --- Activity-Dependent Plasticity with Enhanced Stability ---
                   unwrapped_model = accelerator_arc.unwrap_model(ctm_model_arc)
                   clamped_diffusion_loss = torch.clamp(diffusion_loss, -10.0, 10.0)
                   clamped_ce_loss = torch.clamp(ce_loss, -10.0, 10.0)
                   unwrapped_model.ctm_core.apply_activity_plasticity(clamped_diffusion_loss, clamped_ce_loss, norm_mcmc_loss_for_plasticity)
                   optimizer_arc.step()
                   optimizer_arc.zero_grad()
            else:
               total_loss.backward()
               if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                   # --- Enhanced Gradient Clipping ---
                   total_norm = torch.nn.utils.clip_grad_norm_(ctm_model_arc.parameters(), MAX_GRAD_NORM)
                   if total_norm > MAX_GRAD_NORM * 2:
                       print(f"[WARNING] Large gradient norm detected: {total_norm:.4f}")
                   
                   # --- Activity-Dependent Plasticity with Enhanced Stability ---
                   clamped_diffusion_loss = torch.clamp(diffusion_loss, -10.0, 10.0)
                   clamped_ce_loss = torch.clamp(ce_loss, -10.0, 10.0)
                   ctm_model_arc.ctm_core.apply_activity_plasticity(clamped_diffusion_loss, clamped_ce_loss, norm_mcmc_loss_for_plasticity)
                   optimizer_arc.step()
                   optimizer_arc.zero_grad()
            
            total_arc_loss += total_loss.item()
            processed_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS_ARC}], Batch [{batch_idx+1}/{len(arc_train_loader)}], Loss: {total_loss.item():.4f}")
        
        avg_epoch_loss = total_arc_loss / processed_batches if processed_batches > 0 else 0
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_ARC}] completed. Average Loss: {avg_epoch_loss:.4f}")

        # === SAVE ONLY ON RANK 0 ===
        rank, world_size = get_rank_debug()
        if rank == 0 and CHECKPOINT_DIR_ARC:
            model_to_save_ctm = accelerator_arc.unwrap_model(ctm_model_arc) if accelerator_arc else ctm_model_arc
            model_to_save_head = accelerator_arc.unwrap_model(arc_output_head) if accelerator_arc else arc_output_head

            # Check if DeepSpeed is used and the model is wrapped
            if hasattr(model_to_save_ctm, 'zero_optimization') and hasattr(model_to_save_ctm, 'module'):
                print("✓ Using DeepSpeed consolidated state_dict for CTM model")
                ctm_state_dict = model_to_save_ctm._zero3_consolidated_16bit_state_dict()
            else:
                ctm_state_dict = model_to_save_ctm.state_dict()

            if hasattr(model_to_save_head, 'zero_optimization') and hasattr(model_to_save_head, 'module'):
                print("✓ Using DeepSpeed consolidated state_dict for ARC head")
                head_state_dict = model_to_save_head._zero3_consolidated_16bit_state_dict()
            else:
                head_state_dict = model_to_save_head.state_dict()

            # Save model weights with safetensors
            save_file(ctm_state_dict, os.path.join(CHECKPOINT_DIR_ARC, f"ctm_model_arc_epoch_{epoch+1}.safetensors"))
            save_file(head_state_dict, os.path.join(CHECKPOINT_DIR_ARC, f"arc_output_head_epoch_{epoch+1}.safetensors"))

            # Save optimizer (use torch.save, not supported by safetensors)
            torch.save(optimizer_arc.state_dict(), os.path.join(CHECKPOINT_DIR_ARC, f"optimizer_arc_epoch_{epoch+1}.pt"))

            print(f"✓ Checkpoint saved for epoch {epoch+1} on rank {rank} to {CHECKPOINT_DIR_ARC}")

    print("\n🎉 ARC-AGI-2 Meta-Learning Training Phase Completed!")