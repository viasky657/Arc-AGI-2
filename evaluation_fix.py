import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import json
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

# Setup module paths based on user-provided successful import logic
print("--- Setting up module paths ---")
# Get the absolute path to the project root
project_root = '/workspaces/Arc-AGI-2'
# Define the path to the 'contineous-thought-machines' directory
module_path = os.path.join(project_root, 'contineous-thought-machines')

if module_path not in sys.path:
    sys.path.append(module_path)
    print(f"Added to sys.path: {module_path}")

try:
    from safetensors.torch import load_file
except ImportError:
    print("Warning: safetensors not found. Loading .safetensors will fail.")
    def load_file(path, device="cpu"):
        raise ImportError(f"safetensors is not installed, cannot load {path}")

import importlib.util

# --- Statically Importing EnhancedCTMDiffusion model ---
print("\n--- Statically importing EnhancedCTMDiffusion model ---")
EnhancedCTMDiffusion = None
try:
    from models.ctm_Diffusion_NEWNEW import EnhancedCTMDiffusion
    print(" -> Successfully imported EnhancedCTMDiffusion from models package.")
except ImportError as e_direct:
    print(f"FATAL: Import from models package failed. Last error: {e_direct}")
    EnhancedCTMDiffusion = None # Ensure it's None on failure

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    print("Warning: Hugging Face Accelerate not found. Will run on a single device.")
    ACCELERATE_AVAILABLE = False
    Accelerator = None

# --- Constants and Configuration ---
# These are gathered from your setup script to make this file runnable
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GRID_SIZE = (30, 30)
PADDING_VALUE = -1 # A common padding value for ARC
ARC_INPUT_FLAT_DIM = MAX_GRID_SIZE[0] * MAX_GRID_SIZE[1]
MAX_SEQUENCE_LENGTH = 8192
PADDING_BYTE_VALUE = 0
NUM_ARC_SYMBOLS = 10 # 0-9
LEARNING_RATE = 1e-4

# --- Your Provided Setup Code ---

# ## Data Handling ##
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
        return byte_sequence[:max_len]
    return byte_sequence + bytes([pad_value] * padding_len)

class NewCustomARCGridDataset(Dataset):
    def __init__(self, data_dir, max_grid_size=MAX_GRID_SIZE, padding_value=PADDING_VALUE):
        self.data_dir = data_dir
        self.task_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".json"):
                    self.task_files.append(os.path.join(root, file))
        self.max_grid_size = max_grid_size
        self.padding_value = padding_value
        self.tasks = [json.load(open(f)) for f in self.task_files]
        print(f"Loaded {len(self.tasks)} tasks from {data_dir} (recursively).")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task_data = self.tasks[idx]
        processed_task = {'train': [], 'test': [], 'id': os.path.basename(self.task_files[idx])}
        for pair_type in ['train', 'test']:
            for item in task_data.get(pair_type, []):
                input_grid = item['input']
                output_grid = item['output']
                original_input_dims = (len(input_grid), len(input_grid[0]) if input_grid else 0)
                original_output_dims = (len(output_grid), len(output_grid[0]) if output_grid else 0)
                padded_input = pad_grid(input_grid, self.max_grid_size, self.padding_value)
                padded_output = pad_grid(output_grid, self.max_grid_size, self.padding_value)
                processed_task[pair_type].append({
                    'input': torch.from_numpy(padded_input).long(),
                    'output': torch.from_numpy(padded_output).long(),
                    'original_input_dims': original_input_dims,
                    'original_output_dims': original_output_dims
                })
        return processed_task

def collate_fn_new_custom_arc_eval(batch_of_tasks):
    # This simplified collate is for evaluation (batch size=1)
    return batch_of_tasks[0]

# --- Model Configuration ---
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
    entropy_model_byte_vocab_size: int = 256
    entropy_model_embedding_dim: int = 64
    entropy_model_hidden_dim: int = 128
    entropy_model_num_layers: int = 1
    entropy_model_dropout: float = 0.1
    entropy_model_loss_weight: float = 0.1
    ctm_iterations: int = 5
    ctm_d_model: int = 512
    ctm_input_dim: int = 256
    ctm_heads: int = 8
    ctm_out_dims: int = 512
    ctm_neuron_select_type: str = 'bio_multi_objective'
    attention_type: str = "subquadratic"
    positional_embedding_type: Optional[str] = 'multi-learnable-fourier'
    reshape_patch_sequence_to_grid: bool = True
    patch_grid_width: Optional[int] = 16
    enable_pipeline_parallelism: bool = False # Simplified for eval script
    num_outputs: int = 1
    output_dims: List[int] = field(default_factory=lambda: [512])
    vocab_size: Optional[int] = None
    # Add other fields from your config here if they cause `__post_init__` errors
    def __post_init__(self):
        if len(self.output_dims) != self.num_outputs:
            raise ValueError("output_dims length must match num_outputs")

config_arc_diffusion = EnhancedCTMConfig()

# --- Model and Dataloader Initialization ---
print("--- Initializing Models for Evaluation ---")
ctm_model_arc = None
arc_output_head = None
if EnhancedCTMDiffusion:
    ctm_model_arc = EnhancedCTMDiffusion(config=config_arc_diffusion)
    arc_output_head = nn.Linear(config_arc_diffusion.output_dims[0], ARC_INPUT_FLAT_DIM * NUM_ARC_SYMBOLS)
    print("✓ Real models instantiated.")
    # Move models to the correct device immediately after instantiation
    ctm_model_arc.to(device)
    arc_output_head.to(device)
else:
    print("FATAL: Cannot proceed without EnhancedCTMDiffusion class. The script cannot continue.")
    # Using raise instead of exit() to avoid killing the kernel
    raise ImportError("FATAL: Cannot proceed without EnhancedCTMDiffusion class.")

def find_directory(start_path, dir_name):
    """Recursively finds a directory by name."""
    for root, dirs, _ in os.walk(start_path):
        if dir_name in dirs:
            found_path = os.path.join(root, dir_name)
            print(f"Found '{dir_name}' directory at: {found_path}")
            return found_path
    return None

print("\n--- Searching for evaluation and checkpoint directories ---")
# Path to ARC evaluation tasks
ARC_EVAL_DIR_SEARCHED = find_directory(".", "evaluation")
# Path to CTM checkpoints
CHECKPOINT_DIR_ARC_SEARCHED = find_directory(".", "ctm_arc_agi_2_enhanced_diffusion")

ARC_EVAL_DIR = ARC_EVAL_DIR_SEARCHED if ARC_EVAL_DIR_SEARCHED else "contineous-thought-machines/data/evaluation"
CHECKPOINT_DIR_ARC = CHECKPOINT_DIR_ARC_SEARCHED if CHECKPOINT_DIR_ARC_SEARCHED else os.path.join("checkpoints", "ctm_arc_agi_2_enhanced_diffusion")

if not ARC_EVAL_DIR_SEARCHED:
    print(f"-> Evaluation directory not found dynamically, using fallback: '{ARC_EVAL_DIR}'")
if not CHECKPOINT_DIR_ARC_SEARCHED:
    print(f"-> Checkpoint directory not found dynamically, using fallback: '{CHECKPOINT_DIR_ARC}'")

NUM_EPOCHS_ARC = 20

if os.path.exists(ARC_EVAL_DIR):
    arc_eval_dataset = NewCustomARCGridDataset(ARC_EVAL_DIR)
    arc_eval_loader = DataLoader(arc_eval_dataset, batch_size=1, collate_fn=collate_fn_new_custom_arc_eval)
else:
    print(f"⚠️  Evaluation directory not found at '{ARC_EVAL_DIR}'. Using empty dataloader.")
    arc_eval_loader = []

# --- Main Evaluation Logic ---
print("\n" + "="*60)
print(f"🔬 STARTING ARC-AGI-2 Evaluation on device '{device}'")
print("="*60 + "\n")

if not all([ctm_model_arc, arc_output_head, arc_eval_loader]):
     print("⚠️ Skipping evaluation due to missing components.")
else:
    latest_epoch = NUM_EPOCHS_ARC
    ctm_checkpoint_path_eval = os.path.join(CHECKPOINT_DIR_ARC, f"ctm_model_arc_epoch_{latest_epoch}.safetensors")
    head_checkpoint_path_eval = os.path.join(CHECKPOINT_DIR_ARC, f"arc_output_head_epoch_{latest_epoch}.safetensors")

    try:
        # Load CTM Model
        if os.path.exists(ctm_checkpoint_path_eval):
            print(f"  > Loading CTM checkpoint from {ctm_checkpoint_path_eval}...")
            # Load state_dict to CPU first, then load into the correctly instantiated model
            state_dict_ctm = load_file(ctm_checkpoint_path_eval, device="cpu")
            ctm_model_arc.load_state_dict(state_dict_ctm, strict=False) # Use strict=False to be more robust
            print(f"✓ Loaded CTM checkpoint from epoch {latest_epoch}.")
        else:
            print(f"⚠️ CTM Checkpoint not found at {ctm_checkpoint_path_eval}.")

        # Load ARC Output Head Model
        if os.path.exists(head_checkpoint_path_eval):
            print(f"  > Loading ARC Output Head checkpoint from {head_checkpoint_path_eval}...")
            state_dict_head = load_file(head_checkpoint_path_eval, device="cpu")
            arc_output_head.load_state_dict(state_dict_head, strict=False)
            print(f"✓ Loaded ARC Output Head checkpoint from epoch {latest_epoch}.")
        else:
            print(f"⚠️ ARC Output Head Checkpoint not found at {head_checkpoint_path_eval}.")

        ctm_model_arc.eval()
        arc_output_head.eval()
        
        total_tasks = 0
        solved_tasks = 0
        
        with torch.inference_mode():
            for task_idx, task_batch in enumerate(arc_eval_loader):
                if not task_batch: continue
                
                current_task_data = task_batch
                total_tasks += 1
                task_solved_overall = True

                if 'test' not in current_task_data or not current_task_data['test']:
                    print(f"Task {task_idx + 1} ({current_task_data.get('id', 'N/A')}): No test cases found. Skipping.")
                    task_solved_overall = False
                    continue

                for test_pair_idx, test_pair in enumerate(current_task_data['test']):
                    input_grid_np_eval = test_pair['input'].cpu().numpy()
                    input_bytes_eval_single = serialize_and_pad_grid(input_grid_np_eval, max_len=MAX_SEQUENCE_LENGTH, pad_value=PADDING_BYTE_VALUE)
                    input_bytes_eval = torch.from_numpy(input_bytes_eval_single).to(torch.uint8).unsqueeze(0).to(device)

                    target_grid_np = test_pair['output'].cpu().numpy()
                    original_dims = test_pair['original_output_dims']

                    test_input_solved = False
                    for trial in range(3):
                        current_batch_size_eval = input_bytes_eval.size(0)
                        eval_timestep = torch.zeros(current_batch_size_eval, device=input_bytes_eval.device).long()

                        # This call assumes ctm_model_arc is on the correct device already
                        eval_model_output_dict = ctm_model_arc(
                            byte_sequence=input_bytes_eval,
                            mode='ctm_controlled_diffusion',
                            target_diffusion_output=None,
                            timestep=eval_timestep,
                            task_name="ARC_AGI_2_EVAL_DIFFUSION"
                        )
                        
                        predicted_byte_sequence = eval_model_output_dict.get('diffusion_output_pred')
                        
                        if predicted_byte_sequence is None:
                            print("Warning: Key 'diffusion_output_pred' not found. Trying 'generated_output'.")
                            predicted_byte_sequence = eval_model_output_dict.get('generated_output')
                        
                        if predicted_byte_sequence is None:
                            print("Warning: Generated output key not found. Using zeros as prediction.")
                            preds_grid = np.zeros(MAX_GRID_SIZE, dtype=int)
                        else:
                            if predicted_byte_sequence.ndim == 1 and current_batch_size_eval == 1:
                                predicted_byte_sequence = predicted_byte_sequence.unsqueeze(0)

                            if predicted_byte_sequence.shape[1] >= ARC_INPUT_FLAT_DIM:
                                preds_flat_bytes = predicted_byte_sequence[0, :ARC_INPUT_FLAT_DIM]
                                preds_grid = preds_flat_bytes.view(MAX_GRID_SIZE).long().cpu().numpy()
                            else:
                                print(f"Warning: Generated byte sequence too short. Using zeros.")
                                preds_grid = np.zeros(MAX_GRID_SIZE, dtype=int)
                        
                        h, w = original_dims
                        final_pred = preds_grid[:h, :w]
                        final_target = target_grid_np[:h, :w]

                        if np.array_equal(final_pred, final_target):
                            test_input_solved = True
                            break

                    if not test_input_solved:
                        task_solved_overall = False
                        break
                
                if task_solved_overall:
                    solved_tasks += 1
                    print(f"  Task {task_idx + 1}/{len(arc_eval_loader)} ({current_task_data.get('id', 'N/A')}): SOLVED")
                else:
                    print(f"  Task {task_idx + 1}/{len(arc_eval_loader)} ({current_task_data.get('id', 'N/A')}): FAILED")
        
        if total_tasks > 0:
            accuracy = (solved_tasks / total_tasks) * 100
            summary = f"ARC-AGI-2 Evaluation Summary:\n  Total tasks evaluated: {total_tasks}\n  Tasks solved: {solved_tasks}\n  Accuracy: {accuracy:.2f}%"
            print(f"\n{summary}")
            with open('arc_agi_2_evaluation_summary.txt', 'w') as f:
                f.write(summary)
        else:
            print("\nARC-AGI-2 Evaluation: No tasks were evaluated.")
            
    except FileNotFoundError as e:
        print(f"❌ Checkpoint file not found: {e}. Please ensure paths are correct.")   
    except Exception as e:
        print(f"❌ Error during ARC-AGI-2 evaluation: {e}")
        traceback.print_exc()
        
    print("\n🔬 ARC-AGI-2 Evaluation Phase Completed.")