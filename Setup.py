
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Optional, Callable, Tuple, Dict, Any, List, Union
from dataclasses import dataclass, field
import math
import numpy as np
import warnings # For ARCGridOutputSpace warnings
import sys
import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from accelerate import Accelerator

WORKSPACE_ROOT = "/workspace/Arc-AGI-2"

ARC_TRAIN_DIR = os.path.join(WORKSPACE_ROOT, "contineous_thought_machines", "data", "training")
ARC_EVAL_DIR = os.path.join(WORKSPACE_ROOT, "contineous_thought_machines", "data", "evaluation")

def find_json_file(filename, search_dir):
    """
    Search for a specific JSON file by filename in a given directory tree.
    """
    for root, _, files in os.walk(search_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def resolve_json_files(directory):
    """
    Collect all JSON files in the given directory.
    If any is missing, try to find it elsewhere in the workspace.
    Returns a list of absolute file paths.
    """
    json_files = []
    # Get all JSON files that exist in the given directory
    for file in os.listdir(directory):
        if file.endswith(".json"):
            abs_path = os.path.join(directory, file)
            if os.path.exists(abs_path):
                json_files.append(abs_path)
            else:
                # Try to find it in the workspace
                print(f"[WARN] File not found in expected path: {abs_path}. Searching workspace...")
                found = find_json_file(file, WORKSPACE_ROOT)
                if found:
                    print(f"[INFO] Found {file} at: {found}")
                    json_files.append(found)
                else:
                    print(f"[ERROR] Could not find {file} anywhere in {WORKSPACE_ROOT}")
    return json_files

# Use the function for both training and evaluation dirs
train_json_files = resolve_json_files(ARC_TRAIN_DIR)
eval_json_files = resolve_json_files(ARC_EVAL_DIR)

print(f"✅ Found {len(train_json_files)} training JSON files.")
print(f"✅ Found {len(eval_json_files)} evaluation JSON files.")

# Example: show first few
print("Training files:", train_json_files[:3])
print("Evaluation files:", eval_json_files[:3])


MAX_GRID_SIZE = (30, 30)
NUM_ARC_SYMBOLS = 10
PADDING_VALUE = -1 # A value not in 0-9 to be ignored by the loss function

# Configuration for ARC-AGI-2 Training (shared constants)
ARC_INPUT_FLAT_DIM = MAX_GRID_SIZE[0] * MAX_GRID_SIZE[1]

print(f"Using MAX_GRID_SIZE: {MAX_GRID_SIZE}")
print(f"Using NUM_ARC_SYMBOLS: {NUM_ARC_SYMBOLS}")
print(f"Using ARC_INPUT_FLAT_DIM: {ARC_INPUT_FLAT_DIM}") 

# Ensure the workspace root is in sys.path for correct module resolution.
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)
    print(f"[INFO] Added workspace root to sys.path: {WORKSPACE_ROOT}")

# --- Constants and Configs ---
MAX_GRID_SIZE = (30, 30)
PADDING_VALUE = -1
NUM_ARC_SYMBOLS = 10
ARC_INPUT_FLAT_DIM = MAX_GRID_SIZE[0] * MAX_GRID_SIZE[1]
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = "checkpoints"
ARC_TRAIN_DIR = "/workspace/Arc-AGI-2/contineous_thought_machines/data/training" #Training Dataset Directory

ACCELERATE_AVAILABLE = True
try:
    from accelerate import Accelerator
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None

# Check for xformers
XFORMERS_AVAILABLE = False
if device == "cuda":
    try:
        import xformers
        XFORMERS_AVAILABLE = True
    except ImportError:
        pass

# Check for torch.compile
TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile')

# Check for deepspeed
DEEPSPEED_AVAILABLE = False
if device == "cuda":
    try:
        import deepspeed
        DEEPSPEED_AVAILABLE = True
    except ImportError:
        pass

# A reasonable default for dataloader config
OPTIMIZED_DATALOADER_CONFIG = {
    "num_workers": 4,
    "pin_memory": True,
    "prefetch_factor": 2
} if torch.cuda.is_available() else {}

# --- Context: 2D Grid Padding (from original code) ---
# This function handles padding at the 2D grid level, before serialization.
def pad_grid(grid_list, max_dims, pad_value):
    """Pads a 2D grid to specified maximum dimensions."""
    grid_np = np.array(grid_list, dtype=np.int32)
    padded_grid = np.full(max_dims, pad_value, dtype=np.int32)
    h, w = grid_np.shape
    padded_grid[:h, :w] = grid_np
    return padded_grid

# --- Fix: Byte Sequence Padding for the Model --- #
# According to the model explanation, the key step is to pad the *serialized byte sequence*
# to `config.max_sequence_length`. The function below implements this logic.

# Define the model's expected input dimension from the configuration.
MAX_SEQUENCE_LENGTH = 8192
PADDING_BYTE_VALUE = 0

def serialize_and_pad_grid(grid, max_len=MAX_SEQUENCE_LENGTH, pad_value=PADDING_BYTE_VALUE):
    """
    Serializes a grid into a byte sequence and pads it to a fixed length.

    This function implements the required padding logic for the LearnedBytePatcherEncoder.
    It takes a grid, converts it to a flat byte sequence, and then pads or truncates
    it to `max_sequence_length` (8192 bytes), ensuring a fixed-size input for the model.
    
    Args:
        grid (list or np.ndarray): The input ARC grid.
        max_len (int): The target length for the byte sequence, corresponding to
                       `config.max_sequence_length`.
        pad_value (int): The byte value to use for padding (0-255).

    Returns:
        bytes: The padded byte sequence of length `max_len`.
    """
    # Convert the grid to a NumPy array of single bytes (uint8) and flatten it.
    # ARC values (0-9) fit perfectly within a single byte.
    flat_array = np.array(grid, dtype=np.uint8).flatten()

    # Serialize the flattened array into a raw byte sequence.
    byte_sequence = flat_array.tobytes()

    # Calculate the number of padding bytes needed.
    padding_len = max_len - len(byte_sequence)

    if padding_len < 0:
        # If the original sequence is too long, truncate it.
        padded_sequence = byte_sequence[:max_len]
    else:
        # If the sequence is shorter, create padding and append it.
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
            print(f"NewCustomARCGridDataset Warning: No JSON files found in {data_dir}. Attempting fallback search.")
            base_dir = '/workspace/Arc-AGI-2'
            self.task_files = []
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith('.json'):
                        self.task_files.append(os.path.join(root, file))
            if self.task_files:
                print(f"Found {len(self.task_files)} JSON files via fallback search in {base_dir}")
            else:
                print(f"No JSON files found via fallback search in {base_dir}. Dataset will be empty.")
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

        # Process 'train' pairs from the task
        for train_pair in task.get('train', []):
            if not isinstance(train_pair, dict) or 'input' not in train_pair or 'output' not in train_pair:
                continue

            # train_pair['input'] and train_pair['output'] are already padded 2D LongTensors from NewCustomARCGridDataset
            input_grid_np = train_pair['input'].numpy() # Convert to numpy for serialize_and_pad_grid
            target_grid_np = train_pair['output'].numpy()

            # 1. Create input_byte_sequences (uint8)
            input_bytes = serialize_and_pad_grid(input_grid_np, max_len=MAX_SEQUENCE_LENGTH, pad_value=PADDING_BYTE_VALUE)
            input_byte_sequences_list.append(torch.tensor(list(input_bytes), dtype=torch.uint8))

            # 2. Create target_byte_sequences_for_diffusion (uint8)
            target_bytes_for_diffusion = serialize_and_pad_grid(target_grid_np, max_len=MAX_SEQUENCE_LENGTH, pad_value=PADDING_BYTE_VALUE)
            target_byte_sequences_for_diffusion_list.append(torch.tensor(list(target_bytes_for_diffusion), dtype=torch.uint8))

            # 3. Keep original_target_grids_for_ce_loss (long tensor, flattened)
            original_target_grids_for_ce_loss_list.append(train_pair['output'].view(-1)) # Flattened LongTensor
            
    if not input_byte_sequences_list:
        return {
            'input_byte_sequences': torch.empty(0, MAX_SEQUENCE_LENGTH, dtype=torch.uint8),
            'target_byte_sequences_for_diffusion': torch.empty(0, MAX_SEQUENCE_LENGTH, dtype=torch.uint8),
            'original_target_grids_for_ce_loss': torch.empty(0, ARC_INPUT_FLAT_DIM, dtype=torch.long),
        }

    # Stack all collected tensors
    final_input_byte_sequences = torch.stack(input_byte_sequences_list)
    final_target_byte_sequences_for_diffusion = torch.stack(target_byte_sequences_for_diffusion_list)
    final_original_target_grids_for_ce_loss = torch.stack(original_target_grids_for_ce_loss_list)
    
    return {
        'input_byte_sequences': final_input_byte_sequences,
        'target_byte_sequences_for_diffusion': final_target_byte_sequences_for_diffusion,
        'original_target_grids_for_ce_loss': final_original_target_grids_for_ce_loss,
    }

# --- ARC Training Setup ---
ARC_OUTPUT_HEAD_DIM = ARC_INPUT_FLAT_DIM * NUM_ARC_SYMBOLS
ARC_TASK_ID = 3
print(f"ARC Output Head Dim: {ARC_OUTPUT_HEAD_DIM}")

ctm_model_arc, optimizer_arc, accelerator_arc = None, None, None

print("\n-----------------------------------------------------------------------------")
print("Initializing Configuration for Integrated Diffusion CTM")
print("-----------------------------------------------------------------------------")
print(f"Using device: {device}")
if device == "cuda":
    print("✅ Mixed precision training enabled (BF16) - Expected ~2x speedup")

print("\n🚀 OPTIMIZATION STATUS:")
print(f"  ⚡ torch.compile: {'✅' if TORCH_COMPILE_AVAILABLE else '❌'}")
print(f"  📈 Accelerate: {'✅' if ACCELERATE_AVAILABLE else '❌'}")
print(f"  ⚡ xFormers: {'✅' if XFORMERS_AVAILABLE else '❌'}")
print(f"  ⚡ Deepspeed: {'✅' if DEEPSPEED_AVAILABLE else '❌'}")

# From contineous_thought_machines/models/constants.py
VALID_NEURON_SELECT_TYPES = [
    'first-last', 'random', 'random-pairing',  # Legacy
    # Biologically-inspired types
    'bio_hebbian', 'bio_plasticity', 'bio_competitive', 'bio_homeostatic',
    'bio_evolutionary', 'bio_stdp', 'bio_criticality', 'bio_multi_objective',
    # Hybrid approaches
    'adaptive_random', 'performance_guided', 'task_aware'
]

VALID_POSITIONAL_EMBEDDING_TYPES = [
    'learnable-fourier', 'multi-learnable-fourier',
    'custom-rotational'
]

# From contineous_thought_machines/models/ctm_Diffusion_NEWNEW.py
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union, Any, List
import math

from contineous_thought_machines.models.ctm_Diffusion_NEWNEW import EnhancedCTMDiffusion
from contineous_thought_machines.models.ctm_components import EnhancedCTMConfig

# --- Model Configuration ---
config_arc_diffusion = EnhancedCTMConfig(
    d_model=512,
    n_heads=8,
    n_layers=24,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
    dropout=0.1,
    use_dynamic_entropy_patcher=True,
    patch_embedding_dim=256,
    patch_grid_width=16,
    patch_encoder_cnn_channels=64,
    entropy_patcher_threshold_type="global",
    entropy_patcher_global_threshold=0.75,
    entropy_patcher_relative_threshold=0.1,
    entropy_patcher_min_patch_size=4,
    entropy_patcher_max_patch_size=128,
    entropy_model_byte_vocab_size=256,
    entropy_model_embedding_dim=64,
    entropy_model_hidden_dim=128,
    entropy_model_num_layers=1,
    entropy_model_dropout=0.1,
    entropy_model_loss_weight=0.1,
    ctm_input_dim=256,
    ctm_d_model=512,
    ctm_iterations=5,
    ctm_heads=8,
    ctm_out_dims=512,
    ctm_neuron_select_type='bio_multi_objective',
    positional_embedding_type='multi-learnable-fourier',
    positional_embedding_dim=None,
    reshape_patch_sequence_to_grid=True,
    enable_pipeline_parallelism=True,
    pipeline_stages=4,
    pipeline_overlap_ratio=0.7,
    enable_adaptive_batching=True,
    initial_batch_size=32,
    min_batch_size=8,
    max_batch_size=256,
    batch_adaptation_frequency=100,
    memory_threshold_high=0.85,
    memory_threshold_low=0.6,
    enable_smart_sampling=True,
    sample_importance_weight=0.6,
    sample_diversity_weight=0.4,
    initial_sample_ratio=0.3,
    complexity_analysis_enabled=True,
    num_inputs=1,
    num_outputs=1,
    output_dims=[64],
    ssl_dim=128,
    ssl_weight=0.1,
    ssl_temperature=0.07,
    ssl_noise_std=0.1,
    use_spatial=False,
    use_wina_attention=True,
    max_tasks=50,
    diffusion_steps=1000,
    ctm_diffusion_coupling_strength=0.8,
    vocab_size=None,
    output_audio_bytes=True,
    unet_input_feature_dim=MAX_SEQUENCE_LENGTH // 4, # Calculated based on float32 audio
    local_hebbian_loss_weight=0.01,
    enable_consciousness_controller=True,
    consciousness_max_attention_steps=100,
    use_hrm_core=True,
    attention_type="WINA",
    inferred_task_latent_dim=512, #This does nothing in the model training but is included in a placeholder to avoid possible errors with initializing Torch for training.
    ctm_use_qat=True,
    ctm_adaptive_quantization=True,
    ctm_quant_min_bits=2,
    ctm_quant_max_bits=8,
    ctm_quant_policy_search=True,
    ctm_selective_quantization=True
)
print("✓ EnhancedCTMConfig for ARC (config_arc_diffusion) created.")
    
if 'EnhancedCTMDiffusion' in globals() and EnhancedCTMDiffusion is not None:
    ctm_model_arc = EnhancedCTMDiffusion(config=config_arc_diffusion).to(device)
    print("✓ EnhancedCTMDiffusion model for ARC (ctm_model_arc) initialized.")

    # Prepare model for QAT
    ctm_model_arc.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(ctm_model_arc, inplace=True)
    print("✓ Model prepared for Quantization-Aware Training (QAT).")

    # The new EnhancedCTMDiffusion model is end-to-end and does not require an external output head.
    print("✓ ARC Output Head is disabled as it's not needed for the new model.")

    # MCMC integration is disabled as per new model requirements.
    
    arc_trainable_params = list(ctm_model_arc.parameters()) # EnhancedCTMDiffusion parameters

    optimizer_arc = optim.AdamW([p for p in arc_trainable_params if p.requires_grad], lr=LEARNING_RATE, weight_decay=1e-4)
    
    if ACCELERATE_AVAILABLE:
        accelerator_arc = Accelerator()
        # Only the main model and optimizer need to be prepared.
        ctm_model_arc, optimizer_arc = accelerator_arc.prepare(ctm_model_arc, optimizer_arc)
        print("✓ ARC models (EnhancedCTMDiffusion) and optimizer prepared with Accelerate.")
else:
    print("⚠️ EnhancedCTMDiffusion model or its config for ARC-AGI-2 could not be initialized. Check imports.")

CHECKPOINT_DIR_ARC = os.path.join(CHECKPOINT_DIR, "ctm_arc_agi_2_enhanced_diffusion") # New checkpoint dir
os.makedirs(CHECKPOINT_DIR_ARC, exist_ok=True)
print(f"ARC Checkpoints will be saved to: {CHECKPOINT_DIR_ARC}")

NUM_EPOCHS_ARC = 20
ARC_BATCH_SIZE = 16

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

# The CE loss criterion is no longer needed as the model calculates its own loss.
print("\n✓ ARC-AGI-2 Setup Complete.")
