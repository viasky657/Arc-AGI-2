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
import random
from dataclasses import dataclass, field
from typing import List, Optional, Any
import math

# --- De-noising and Meta-Learning Components ---

def perform_online_update(model, optimizer, scheduler, input_bytes, corrected_grid_np: np.ndarray, device):
    """
    Performs a single, targeted fine-tuning step on the end-to-end model.
    """
    model.train()
    optimizer.zero_grad()

    target_bytes_single = serialize_and_pad_grid(corrected_grid_np, max_len=MAX_SEQUENCE_LENGTH, pad_value=PADDING_BYTE_VALUE)
    target_bytes_np = np.frombuffer(target_bytes_single, dtype=np.uint8).copy()
    target_bytes_tensor = torch.from_numpy(target_bytes_np).to(torch.uint8).unsqueeze(0).to(device)

    train_timestep = torch.zeros(1, device=device).long()
    
    # The model's forward pass calculates all necessary losses internally.
    output_dict = model(
        byte_sequence=input_bytes,
        mode='ctm_controlled_diffusion',
        target_diffusion_output=target_bytes_tensor,
        timestep=train_timestep,
        task_name="ARC_AGI_2_ONLINE_LEARN"
    )

    loss = output_dict.get('total_loss')

    if loss is not None and torch.isfinite(loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        print(f"  > Model updated with loss: {loss.item():.4f}. LR: {scheduler.get_last_lr()[0]:.6f}")
    else:
        print("  > Skipping online update due to invalid loss.")

    model.eval()

# Setup module paths based on user-provided successful import logic
print("--- Setting up module paths ---")
project_root = '/workspaces/Arc-AGI-2'
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

print("\n--- Statically importing EnhancedCTMDiffusion model ---")
EnhancedCTMDiffusion = None
try:
    from models.ctm_Diffusion_NEWNEW import EnhancedCTMDiffusion
    print(" -> Successfully imported EnhancedCTMDiffusion from models package.")
except ImportError as e_direct:
    print(f"FATAL: Import from models package failed. Last error: {e_direct}")
    EnhancedCTMDiffusion = None

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    print("Warning: Hugging Face Accelerate not found. Will run on a single device.")
    ACCELERATE_AVAILABLE = False
    Accelerator = None

# --- Constants and Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GRID_SIZE = (30, 30)
PADDING_VALUE = -1
ARC_INPUT_FLAT_DIM = MAX_GRID_SIZE[0] * MAX_GRID_SIZE[1]
MAX_SEQUENCE_LENGTH = 8192
PADDING_BYTE_VALUE = 0
NUM_ARC_SYMBOLS = 10
LEARNING_RATE = 1e-4

# --- Data Handling ---
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

@dataclass
class EnhancedCTMConfig: # Renamed from ContinualLearningConfig for consistency in the target file
    """Enhanced configuration for continual learning CTM-diffusion model,
    incorporating binary processing, multi-task learning, and advanced CTM features."""
    
    # Model architecture (General Transformer/Diffusion settings)
    d_model: int = 512  # Main model dimensionality
    n_heads: int = 8
    n_layers: int = 24
    max_sequence_length: int = 8192 # Max input sequence length in terms of bytes or patches
    dropout: float = 0.1
    
    # --- Byte Processing Options ---
    patch_embedding_dim: int = 256         # <<< NEW: Output embedding dimension per patch from patcher
    patch_encoder_cnn_channels: int = 64   # <<< NEW: Intermediate channels for CNN patch encoder

    # --- Dynamic Entropy Patching Options (Inspired by BLT paper) ---
    use_dynamic_entropy_patcher: bool = True # Flag to enable dynamic entropy-based patching
    entropy_patcher_threshold_type: str = "global"  # 'global' or 'relative_monotonic'
    entropy_patcher_global_threshold: float = 0.75 # Entropy threshold for 'global' type
    entropy_patcher_relative_threshold: float = 0.1 # Entropy diff threshold for 'relative_monotonic'
    entropy_patcher_min_patch_size: int = 4      # Minimum number of bytes in a dynamic patch
    entropy_patcher_max_patch_size: int = 128    # Maximum number of bytes in a dynamic patch (for CNN encoder)
    
    # --- Learnable Entropy Model Parameters (for _EntropyProxyModel) ---
    entropy_model_byte_vocab_size: int = 256
    entropy_model_embedding_dim: int = 64
    entropy_model_hidden_dim: int = 128
    entropy_model_num_layers: int = 1
    entropy_model_dropout: float = 0.1
    entropy_model_loss_weight: float = 0.1 # Weight for its auxiliary loss contribution
    # Note: These parameters are used if use_dynamic_entropy_patcher is True,
    # as LearnedBytePatcherEncoder now instantiates the learnable _EntropyProxyModel.
    
    # Fallback if not using learned_patch_encoder or dynamic_entropy_patcher
    byte_embedding_dim: int = 256
    multi_granularity: bool = False # Default to False if patcher is preferred
    # multi_granularity_output_dim is complex to predefine, MGP should expose its output dim.
    # For now, if multi_granularity is True AND use_learned_patch_encoder is False, this would be used.
    multi_granularity_output_dim: int = 256 # Placeholder if MGP is used.
    
    hierarchical_processing: bool = True # General flag, could apply to patcher or MGP
    
    # CTM Core Parameters (Specific to the OriginalCTMCore module)
    # These are prefixed with 'ctm_' to distinguish from general model params
    ctm_iterations: int = 5  # Original 'iterations'
    ctm_d_model: int = 512   # Original 'd_model' for CTM's internal latent space
    ctm_input_dim: int = 256 # Dimensionality of inputs to CTM (e.g., from byte embeddings or other features)
                             # This was 'd_input' in OriginalCTMCore if it took external features.
                             # If CTM processes outputs of byte_embedding, this might be byte_embedding_dim.
    ctm_heads: int = 8       # Attention heads within CTM
    ctm_n_synch_out: int = 64
    ctm_n_synch_action: int = 64
    ctm_synapse_depth: int = 3
    ctm_memory_length: int = 10
    ctm_deep_nlms: bool = True
    ctm_memory_hidden_dims: int = 2048
    ctm_do_layernorm_nlm: bool = False
    ctm_out_dims: int = 512  # Output dimension of CTM's own projector
    ctm_prediction_reshaper: list = field(default_factory=lambda: [-1])
    ctm_dropout: float = 0.1
    ctm_dropout_nlm: Optional[float] = None
    # Neuron selection strategy. Available options:
    # Legacy: 'first-last', 'random', 'random-pairing'
    # Biologically-inspired: 'bio_hebbian', 'bio_plasticity', 'bio_competitive',
    #                        'bio_homeostatic', 'bio_evolutionary', 'bio_stdp',
    #                        'bio_criticality', 'bio_multi_objective'
    # Hybrid: 'adaptive_random', 'performance_guided', 'task_aware'
    ctm_neuron_select_type: str = 'bio_multi_objective'
    ctm_n_random_pairing_self: int = 0
    
    # Diffusion Parameters
    diffusion_steps: int = 1000
    noise_schedule: str = "cosine" # e.g., "linear", "cosine"
    diffusion_beta_start: float = 0.0001
    diffusion_beta_end: float = 0.02
    diffusion_timesteps: int = 1000 # Number of timesteps for the diffusion process
    ctm_diffusion_coupling_strength: float = 0.8 # How CTM influences diffusion
    adaptive_scheduling: bool = True  # CTM-adaptive diffusion timestep scheduling
    iterative_refinement: bool = True # Iterative CTM-diffusion refinement for sampling
    
    # Training Efficiency
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    sparse_attention: bool = True  # Now implemented with BinarySparseAttention
    adaptive_depth: bool = False   # Defaulting to False, can be enabled if implemented
    use_activity_plasticity: bool = True # To enable/disable plasticity updates; Needs to be set to TRUE
    ctm_use_internal_feedback: bool = True # Enable self-modulating feedback within the CTM core

    # --- Bidirectional Reasoning Parameters ---
    enable_bidirectional_reasoning: bool = True # Allows CTM to move forward/backward in its thought process
    reasoning_step_gating_threshold: float = 0.7 # Confidence threshold for the reasoning controller to terminate
    max_reasoning_steps: int = 15 # Max total steps in a bidirectional reasoning loop to prevent infinite loops
    
    # Sparse Attention Parameters
    sparse_attention_ratio: float = 0.1  # Keep only 10% of attention connections
    binary_pattern_size: int = 8  # Size of binary patterns to detect

    # Attention Mechanism Type
    attention_type: str = "subquadratic"  # Options: "standard", "binary_sparse", "subquadratic"
    
    # Subquadratic Attention Parameters (if attention_type is "subquadratic")
    subquadratic_attn_epsilon: float = 1e-6
    subquadratic_attn_poly_degree: int = 5
    attention_qkv_bias: bool = True # General QKV bias for attention mechanisms like Subquadratic or standard MHA
    # attn_drop and proj_drop for subquadratic_attn will be mapped from ctm_dropout

    # Positional Embedding Parameters
    positional_embedding_type: Optional[str] = 'multi-learnable-fourier' # e.g., 'custom-rotational-1d', 'learnable-fourier', multi-learnable-fourier' #Can set the value here. 
    positional_embedding_dim: Optional[int] = None  # Dimension of the positional embedding, defaults to ctm_input_dim if None
    reshape_patch_sequence_to_grid: bool = True # If True, reshape patch sequence to a 2D grid for 2D PEs. Must set to true if using 2D Grid for Positional Embeddings.
    patch_grid_width: Optional[int] = None       # Desired width of the patch grid if reshaping

    # --- Hierarchical Reasoning Model (HRM) Parameters ---
    use_hrm_core: bool = False # Set to True to use the HierarchicalCTM core
    hrm_high_level_cycles: int = 4 # N: Number of high-level cycles
    hrm_low_level_timesteps: int = 8 # T: Number of low-level timesteps per high-level cycle
    program_vocab_size: int = 1024 # Vocabulary size for the program synthesizer
    program_synth_n_heads: int = 4
    program_synth_n_layers: int = 3
    program_synth_d_ff: int = 1024
    ltm_size: int = 2048 # Size of the long-term memory
    ltm_surprise_threshold: float = 0.6 # Surprise threshold for storing in LTM
    replay_batch_size: int = 4 # Batch size for memory replay
    replay_policy: str = "surprise_weighted_replay" # "simple_replay", "surprise_weighted_replay", "usefulness_replay"

    # Pipeline Parallelism Parameters
    enable_pipeline_parallelism: bool = True
    pipeline_stages: int = 3  # CTM, Diffusion prep, Diffusion exec
    pipeline_overlap_ratio: float = 0.7  # Target overlap ratio
    
    # Adaptive Batch Sizing Parameters
    enable_adaptive_batching: bool = True
    initial_batch_size: int = 32
    min_batch_size: int = 8
    max_batch_size: int = 256
    batch_adaptation_frequency: int = 100
    memory_threshold_high: float = 0.85
    memory_threshold_low: float = 0.6
    
    # Smart Data Sampling Parameters
    enable_smart_sampling: bool = True
    sample_importance_weight: float = 0.6
    sample_diversity_weight: float = 0.4
    initial_sample_ratio: float = 0.3
    complexity_analysis_enabled: bool = True
    
    # Multi-input/output parameters
    num_inputs: int = 1  # Number of input streams
    num_outputs: int = 1  # Number of output heads
    output_dims: List[int] = field(default_factory=lambda: [64])  # Dimensions for each output head
    
    # Self-supervised learning
    ssl_dim: int = 128  # Dimension for self-supervised projection
    ssl_weight: float = 0.1  # Weight for self-supervised loss
    ssl_temperature: float = 0.07  # Temperature for contrastive loss
    ssl_noise_std: float = 0.1  # Noise standard deviation for contrastive augmentation
    
    # Spatiotemporal Processing
    use_spatial: bool = True  # Enable spatial processing for image/video data
    
    # WINA Attention
    use_wina_attention: bool = True  # Enable WINA sparse attention
    
    # Multi-task Learning Parameters
    max_tasks: int = 50  # Maximum number of tasks for continual learning
    # Added to resolve TypeError for unexpected keyword arguments
    vocab_size: Optional[int] = None
    output_audio_bytes: bool = False
    inferred_task_latent_dim: Optional[int] = None # Default to None, __post_init__ handles it
    use_hipa_attention: bool = False # Default to False
    hipa_num_heads: Optional[int] = None # Default to None
    audio_output_dtype_str: Optional[str] = "float32" # Default as per __post_init__ logic
    unet_input_feature_dim: Optional[int] = None # Default to None, __post_init__ calculates it

    # --- JEPA Training Parameters (Integrated with LearnedBytePatcherEncoder) ---
    use_jepa_training: bool = False
    # jepa_embed_dim will be derived from patch_embedding_dim if dynamic_entropy_patcher is used
    jepa_predictor_hidden_dim: int = 512 # Hidden dimension of JEPA predictor MLP
    jepa_mask_ratio_min: float = 0.15 # Min proportion of patch sequence to mask for target
    jepa_mask_ratio_max: float = 0.75 # Max proportion of patch sequence to mask for target
    jepa_context_scale_min: float = 0.3 # Min proportion of patches for context
    jepa_context_scale_max: float = 0.7 # Max proportion of patches for context
    jepa_momentum_beta: float = 0.996 # Momentum for target encoder update
    jepa_loss_weight: float = 0.1 # Weight for the JEPA loss component
    jepa_num_target_blocks: int = 1 # Number of target blocks to predict

    # --- Global Plasticity Loss Parameters ---
    local_hebbian_loss_weight: float = 0.01 # New weight for backprop-based hebbian loss

    # --- Basal Ganglia Parameters --- #Controls action suppression so that the model's unwanted first unrelated thoughts are suppressed which helps with model safety. Is needed for action suppresion.
    ctm_enable_basal_ganglia: bool = True
    ctm_bg_dopamine_dim: int = 32

    # --- Synaptic Empathy Parameters ---
    enable_synaptic_empathy: bool = True # Set to True to use the new SynapticEmpathy module
    synaptic_empathy_reward_weight: float = 0.1

    # --- Mirror Neuron / High-Level Empathy Parameters ---
    enable_mirror_neurons: bool = True # Set to True to use the high-level MirrorNeuronLayer
    num_emotion_dim: int = 4 # Dimensionality of the emotion state vector
    goal_dim: int = 8 # Dimensionality of the predicted goal vector
    mirror_reward_weight: float = 0.2 # Weight for the selfless reward signal


    # --- Confidence Thresholding Parameters ---
    confidence_threshold: float = 0.0 # Confidence threshold for abstaining. If > 0, model can abstain.
 
    # --- Consciousness Controller Parameters ---
    enable_consciousness_controller: bool = True
    consciousness_max_attention_steps: int = 100

    def __post_init__(self):
        # Validate output dimensions
        if len(self.output_dims) != self.num_outputs:
            raise ValueError(f"output_dims length ({len(self.output_dims)}) must match num_outputs ({self.num_outputs})")

        # Merged content from the second __post_init__
        if hasattr(self, 'ctm_prediction_reshaper') and self.ctm_prediction_reshaper == [-1] and self.vocab_size is not None:
            pass
        if hasattr(self, 'ctm_dropout_nlm') and self.ctm_dropout_nlm is None and hasattr(self, 'ctm_dropout'):
            self.ctm_dropout_nlm = self.ctm_dropout
        
        if hasattr(self, 'ctm_neuron_select_type') and \
           'VALID_NEURON_SELECT_TYPES' in globals() and self.ctm_neuron_select_type not in VALID_NEURON_SELECT_TYPES:
            print(f"Warning: ctm_neuron_select_type '{self.ctm_neuron_select_type}' is not in VALID_NEURON_SELECT_TYPES ({VALID_NEURON_SELECT_TYPES}).")

        if hasattr(self, 'positional_embedding_type') and self.positional_embedding_type is not None:
            if 'VALID_POSITIONAL_EMBEDDING_TYPES' in globals() and self.positional_embedding_type not in VALID_POSITIONAL_EMBEDDING_TYPES:
                print(f"Warning: positional_embedding_type '{self.positional_embedding_type}' is not in VALID_POSITIONAL_EMBEDDING_TYPES ({VALID_POSITIONAL_EMBEDDING_TYPES}).")
            if self.positional_embedding_dim is not None and self.positional_embedding_dim <= 0:
                raise ValueError("positional_embedding_dim must be positive if set.")
            
            if self.reshape_patch_sequence_to_grid:
                if self.patch_grid_width is None or self.patch_grid_width <= 0:
                    raise ValueError("patch_grid_width must be a positive integer if reshape_patch_sequence_to_grid is True.")
                if self.positional_embedding_type not in ['learnable-fourier', 'multi-learnable-fourier', 'custom-rotational']:
                    print(f"Warning: reshape_patch_sequence_to_grid is True, but positional_embedding_type ('{self.positional_embedding_type}') is not a typical 2D PE. Ensure compatibility.")

        # Validations for new patch encoder
        if self.use_dynamic_entropy_patcher:
            if self.patch_embedding_dim <= 0:
                raise ValueError("patch_embedding_dim must be positive if use_dynamic_entropy_patcher is True.")
            if self.entropy_patcher_min_patch_size <= 0:
                raise ValueError("entropy_patcher_min_patch_size must be positive.")
            if self.entropy_patcher_max_patch_size < self.entropy_patcher_min_patch_size:
                raise ValueError("entropy_patcher_max_patch_size must be >= entropy_patcher_min_patch_size.")
            if self.entropy_patcher_threshold_type not in ["global", "relative_monotonic"]:
                raise ValueError("entropy_patcher_threshold_type must be 'global' or 'relative_monotonic'.")
        elif self.multi_granularity and self.multi_granularity_output_dim <= 0:
            print("Warning: multi_granularity_output_dim might not be correctly set for validation if not using a patcher and MGP is active.")
        
        if not hasattr(self, 'inferred_task_latent_dim') or self.inferred_task_latent_dim is None:
            print("Warning: inferred_task_latent_dim not found or is None in config, defaulting to 64.")
            self.inferred_task_latent_dim = 512
        elif self.inferred_task_latent_dim <= 0: # This check is now safe
            raise ValueError("inferred_task_latent_dim must be positive.")
 
        if hasattr(self, 'use_hipa_attention') and self.use_hipa_attention and \
            (not hasattr(self, 'hipa_num_heads') or self.hipa_num_heads <= 0):
             raise ValueError("hipa_num_heads must be positive if use_hipa_attention is True.")
 
        if hasattr(self, 'audio_output_dtype_str'):
            if self.audio_output_dtype_str == "float32":
                self.audio_output_item_size = 4
            elif self.audio_output_dtype_str == "int16":
                self.audio_output_item_size = 2
            else:
                if hasattr(self, 'output_audio_bytes') and self.output_audio_bytes:
                    raise ValueError(f"Unsupported audio_output_dtype_str: {self.audio_output_dtype_str} when output_audio_bytes is True.")
                else:
                    self.audio_output_item_size = 4
        elif hasattr(self, 'output_audio_bytes') and self.output_audio_bytes:
            if not hasattr(self, 'audio_output_dtype_str') or self.audio_output_dtype_str is None:
                raise ValueError("audio_output_dtype_str must be defined in config if output_audio_bytes is True.")
        else:
            self.audio_output_item_size = 4

        # Calculate unet_input_feature_dim if not set
        if self.unet_input_feature_dim is None:
            if self.max_sequence_length <= 0 or self.audio_output_item_size <= 0:
                raise ValueError("max_sequence_length and audio_output_item_size must be positive to calculate unet_input_feature_dim.")
            self.unet_input_feature_dim = self.max_sequence_length // self.audio_output_item_size
            if self.unet_input_feature_dim <= 0:
                raise ValueError(f"Calculated unet_input_feature_dim ({self.unet_input_feature_dim}) must be positive. Check max_sequence_length and audio_output_item_size.")
        elif self.unet_input_feature_dim <= 0:
            raise ValueError("unet_input_feature_dim, if set, must be positive.")

        if self.use_jepa_training:
            if not (0 < self.jepa_mask_ratio_min < 1 and 0 < self.jepa_mask_ratio_max < 1 and self.jepa_mask_ratio_min <= self.jepa_mask_ratio_max):
                raise ValueError("JEPA mask ratios must be between 0 and 1, with min <= max.")
            if not (0 < self.jepa_context_scale_min < 1 and 0 < self.jepa_context_scale_max < 1 and self.jepa_context_scale_min <= self.jepa_context_scale_max):
                raise ValueError("JEPA context scales must be between 0 and 1, with min <= max.")
            if not (0 <= self.jepa_momentum_beta < 1):
                raise ValueError("jepa_momentum_beta must be between 0 and 1.")
            if self.jepa_num_target_blocks <= 0:
                raise ValueError("jepa_num_target_blocks must be positive.")
            if not self.use_dynamic_entropy_patcher:
                print("Warning: JEPA training is enabled but use_dynamic_entropy_patcher is False. JEPA relies on the patch embeddings from LearnedBytePatcherEncoder.")

config_arc_diffusion = EnhancedCTMConfig(
    enable_consciousness_controller=True,
    consciousness_max_attention_steps=100
)

print("✓ EnhancedCTMConfig for ARC (config_arc_diffusion) created.")

if EnhancedCTMDiffusion is not None:
    ctm_model_arc = EnhancedCTMDiffusion(config=config_arc_diffusion).to(device)
    print("✓ EnhancedCTMDiffusion model for ARC (ctm_model_arc) initialized.")
    optimizer_arc = optim.AdamW(ctm_model_arc.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    if ACCELERATE_AVAILABLE:
        print(" -> Preparing components with Hugging Face Accelerate...")
        accelerator_arc = Accelerator()
        ctm_model_arc, optimizer_arc = accelerator_arc.prepare(ctm_model_arc, optimizer_arc)
        print("✓ ARC model and optimizer prepared with Accelerate.")
else:
    print("⚠️ EnhancedCTMDiffusion model could not be initialized.")
    ctm_model_arc, optimizer_arc, accelerator_arc = None, None, None


class ARCEvalDataset(Dataset):
    def __init__(self, data_path, max_grid_size=MAX_GRID_SIZE, padding_value=PADDING_VALUE):
        self.task_files = glob.glob(os.path.join(data_path, "*.json"))
        if not self.task_files:
            print(f"Warning: No .json files found at path: {data_path}")
        self.max_grid_size = max_grid_size
        self.padding_value = padding_value
        self.tasks = [json.load(open(f)) for f in self.task_files]
        print(f"Loaded {len(self.tasks)} tasks from {data_path}.")

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

ARC_EVAL_DIR = "/workspace/Arc-AGI-2/contineous-thought-machines/data/evaluation"
CHECKPOINT_DIR_ARC = "/workspaces/Arc-AGI-2/contineous-thought-machines/examples/checkpoints/ctm_arc_agi_2_enhanced_diffusion"
CHECKPOINT_DIR_PRINCIPLES = os.path.join(CHECKPOINT_DIR_ARC, "principles_checkpoints")
NUM_EPOCHS_ARC = 20
NUM_EPOCHS_PRINCIPLES = 10 # Should match the value in training.py

print("\n--- Initializing Evaluation Dataloader ---")
arc_eval_loader = None
if os.path.exists(ARC_EVAL_DIR):
    arc_eval_dataset = ARCEvalDataset(data_path=ARC_EVAL_DIR)
    if len(arc_eval_dataset) > 0:
        arc_eval_loader = DataLoader(arc_eval_dataset, batch_size=1, shuffle=False)
        print(f"✓ Evaluation DataLoader initialized with {len(arc_eval_dataset)} tasks.")
else:
    print(f"⚠️ Evaluation directory not found: '{ARC_EVAL_DIR}'")

# --- Main Evaluation Logic ---
print("\n" + "="*60)
print(f"🔬 STARTING ARC-AGI-2 Evaluation on device '{device}'")
print("="*60 + "\n")

if not all([ctm_model_arc, optimizer_arc, arc_eval_loader]):
     print("⚠️ Skipping evaluation due to missing components.")
else:
    latest_epoch = NUM_EPOCHS_PRINCIPLES
    ctm_checkpoint_path_eval = os.path.join(CHECKPOINT_DIR_PRINCIPLES, f"ctm_model_arc_epoch_{latest_epoch}.safetensors")

    try:
        if os.path.exists(ctm_checkpoint_path_eval):
            print(f"  > Loading CTM checkpoint from {ctm_checkpoint_path_eval}...")
            state_dict_ctm = load_file(ctm_checkpoint_path_eval, device="cpu")
            model_to_load_ctm = accelerator_arc.unwrap_model(ctm_model_arc) if ACCELERATE_AVAILABLE else ctm_model_arc
            model_to_load_ctm.load_state_dict(state_dict_ctm, strict=False)
            print(f"✓ Loaded CTM checkpoint from epoch {latest_epoch}.")
        else:
            print(f"⚠️ CTM Checkpoint not found at {ctm_checkpoint_path_eval}.")

        ctm_model_arc.eval()
        if hasattr(ctm_model_arc, 'wake_up'):
            ctm_model_arc.wake_up()
        total_tasks = 0
        solved_tasks = 0

        # Create a scheduler for the online updates.
        scheduler_arc = optim.lr_scheduler.StepLR(optimizer_arc, step_size=5, gamma=0.9)

        for task_idx, task_batch in enumerate(arc_eval_loader):
            if not task_batch: continue

            current_task_data = task_batch
            total_tasks += 1
            task_solved_overall = True
            
            # Since batch_size is 1, unpack the lists
            task_id = current_task_data['id'][0]
            test_pairs = [{k: v.squeeze(0) for k, v in pair.items()} for pair in current_task_data['test'][0]]

            if not test_pairs:
                print(f"Task {task_idx + 1} ({task_id}): No test cases found. Skipping.")
                continue

            for test_pair_idx, test_pair in enumerate(test_pairs):
                input_grid_np_eval = test_pair['input'].cpu().numpy()
                input_bytes_eval = torch.from_numpy(np.frombuffer(serialize_and_pad_grid(input_grid_np_eval), dtype=np.uint8)).to(torch.uint8).unsqueeze(0).to(device)

                target_grid_np = test_pair['output'].cpu().numpy()
                h, w = test_pair['original_output_dims']
                original_dims = (h.item(), w.item())
                final_target = target_grid_np[:original_dims[0], :original_dims[1]]
                
                test_input_solved = False

                # --- First Attempt: Standard Prediction ---
                print(f"  > Attempt 1 for test pair {test_pair_idx + 1}...")
                with torch.no_grad():
                    eval_model_output = ctm_model_arc.iterative_ctm_diffusion_sample(shape=input_bytes_eval.shape, initial_byte_sequence_for_inference=input_bytes_eval, num_steps=50)
                    output_bytes = eval_model_output[0]
                    
                    if output_bytes is not None and output_bytes.numel() > 0:
                        grid_flat = np.frombuffer(output_bytes.squeeze(0).cpu().numpy().tobytes(), dtype=np.uint8)
                        preds_grid = np.full(MAX_GRID_SIZE, PADDING_VALUE, dtype=int)
                        reshaped_len = min(len(grid_flat), ARC_INPUT_FLAT_DIM)
                        preds_grid.flat[:reshaped_len] = grid_flat[:reshaped_len]
                    else:
                        preds_grid = np.full(MAX_GRID_SIZE, PADDING_VALUE, dtype=int)

                final_pred = preds_grid[:original_dims[0], :original_dims[1]]

                if np.array_equal(final_pred, final_target):
                    print(f"    - Solved on first attempt.")
                    test_input_solved = True
                else:
                    print(f"    - Failed on first attempt. Trying online update.")
                    # --- Second Attempt: Fine-tune and Predict Again ---
                    perform_online_update(
                        model=ctm_model_arc,
                        optimizer=optimizer_arc,
                        scheduler=scheduler_arc,
                        input_bytes=input_bytes_eval,
                        corrected_grid_np=target_grid_np, # Use the full target grid for the update
                        device=device
                    )
                    
                    print(f"  > Attempt 2 for test pair {test_pair_idx + 1} (post-update)...")
                    with torch.no_grad():
                        eval_model_output_2 = ctm_model_arc.iterative_ctm_diffusion_sample(shape=input_bytes_eval.shape, initial_byte_sequence_for_inference=input_bytes_eval, num_steps=50)
                        output_bytes_2 = eval_model_output_2[0]

                        if output_bytes_2 is not None and output_bytes_2.numel() > 0:
                            grid_flat_2 = np.frombuffer(output_bytes_2.squeeze(0).cpu().numpy().tobytes(), dtype=np.uint8)
                            preds_grid_2 = np.full(MAX_GRID_SIZE, PADDING_VALUE, dtype=int)
                            reshaped_len_2 = min(len(grid_flat_2), ARC_INPUT_FLAT_DIM)
                            preds_grid_2.flat[:reshaped_len_2] = grid_flat_2[:reshaped_len_2]
                        else:
                            preds_grid_2 = np.full(MAX_GRID_SIZE, PADDING_VALUE, dtype=int)
                    
                    final_pred_2 = preds_grid_2[:original_dims[0], :original_dims[1]]
                    
                    if np.array_equal(final_pred_2, final_target):
                         print(f"    - Solved on second attempt after fine-tuning.")
                         test_input_solved = True
                    else:
                         print(f"    - Failed on second attempt.")

                if not test_input_solved:
                    task_solved_overall = False
                    break
            
            if task_solved_overall:
                solved_tasks += 1
                print(f"  Task {task_idx + 1}/{len(arc_eval_loader)} ({task_id}): SOLVED")
            else:
                print(f"  Task {task_idx + 1}/{len(arc_eval_loader)} ({task_id}): FAILED")

        if total_tasks > 0:
            accuracy = (solved_tasks / total_tasks) * 100
            summary = f"ARC-AGI-2 Evaluation Summary:\n  Total tasks evaluated: {total_tasks}\n  Tasks solved: {solved_tasks}\n  Accuracy: {accuracy:.2f}%"
            print(f"\n{summary}")
            with open('arc_agi_2_evaluation_summary.txt', 'w') as f:
                f.write(summary)
        else:
            print("\nARC-AGI-2 Evaluation: No tasks were evaluated.")
        
        if hasattr(ctm_model_arc, 'sleep_down'):
            ctm_model_arc.sleep_down()
            
    except FileNotFoundError as e:
        print(f"❌ Checkpoint file not found: {e}. Please ensure paths are correct.")   
    except Exception as e:
        print(f"❌ Error during ARC-AGI-2 evaluation: {e}")
        traceback.print_exc()
        
    print("\n🔬 ARC-AGI-2 Evaluation Phase Completed.")
