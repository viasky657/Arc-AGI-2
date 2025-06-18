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

# A relative path for the evaluation data.
evaluation_relative_path = "contineous-thought-machines/data/evaluation"
# Get the absolute path for the evaluation data
data_path = os.path.abspath(evaluation_relative_path)
print(f"The relative evaluation path is: '{evaluation_relative_path}'")
print(f"The absolute evaluation path is: '{data_path}'")

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
ARC_OUTPUT_HEAD_DIM = ARC_INPUT_FLAT_DIM * NUM_ARC_SYMBOLS
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

# Define EnhancedCTMConfig for ARC with EnhancedCTMDiffusion
# Assuming EnhancedCTMConfig is a defined class and MAX_SEQUENCE_LENGTH is a defined variable
# For example:
# from your_model_library import EnhancedCTMConfig
# MAX_SEQUENCE_LENGTH = 8192

# From contineous-thought-machines/models/constants.py
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
    'custom-rotational', 'custom-rotational-1d'
]

# From contineous-thought-machines/models/ctm_Diffusion_NEWNEW.py
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union, Any, List

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

    # Pipeline Parallelism Parameters
    enable_pipeline_parallelism: bool = True
    pipeline_stages: int = 4  # CTM, MCMC, Diffusion prep, Diffusion exec
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

    # --- Knowledge Store Parameters ---

    def __post_init__(self):
        # Validate output dimensions
        if len(self.output_dims) != self.num_outputs:
            raise ValueError(f"output_dims length ({len(self.output_dims)}) must match num_outputs ({self.num_outputs})")

        # Merged content from the second __post_init__
        if hasattr(self, 'ctm_prediction_reshaper') and self.ctm_prediction_reshaper == [-1] and self.vocab_size is not None:
            pass
        if hasattr(self, 'ctm_dropout_nlm') and self.ctm_dropout_nlm is None and hasattr(self, 'ctm_dropout'):
            self.ctm_dropout_nlm = self.ctm_dropout
        if hasattr(self, 'mcmc_output_space_dim') and self.mcmc_output_space_dim is None and hasattr(self, 'ctm_out_dims'):
            self.mcmc_output_space_dim = self.ctm_out_dims
        
        if hasattr(self, 'ctm_neuron_select_type') and \
           VALID_NEURON_SELECT_TYPES is not None and self.ctm_neuron_select_type not in VALID_NEURON_SELECT_TYPES:
            print(f"Warning: ctm_neuron_select_type '{self.ctm_neuron_select_type}' is not in VALID_NEURON_SELECT_TYPES ({VALID_NEURON_SELECT_TYPES}).")

        if hasattr(self, 'positional_embedding_type') and self.positional_embedding_type is not None:
            if VALID_POSITIONAL_EMBEDDING_TYPES is None: # Fallback if import failed
                print(f"Warning: VALID_POSITIONAL_EMBEDDING_TYPES not available for validation.")
            elif self.positional_embedding_type not in VALID_POSITIONAL_EMBEDDING_TYPES:
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

# Define EnhancedCTMConfig for ARC with EnhancedCTMDiffusion
config_arc_diffusion = EnhancedCTMConfig(
    d_model=512,
    #inferred_task_latent_dim=64, # This line remains commented out
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
    # Parameters for the learnable entropy model within LearnedBytePatcherEncoder
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
    
    # Attention Mechanism Type
    attention_type="subquadratic",  # Options: "standard", "binary_sparse", "subquadratic"
    
    # Subquadratic Attention Parameters
    subquadratic_attn_epsilon=1e-6,
    subquadratic_attn_poly_degree=5,
    attention_qkv_bias=True, # Corrected capitalization
    
    # Positional Embedding Parameters
    positional_embedding_type='multi-learnable-fourier',
    positional_embedding_dim=None,
    reshape_patch_sequence_to_grid=True,
    #patch_grid_width=None, #Already defined in the byte patch section of this config. 

    # Pipeline Parallelism Parameters
    enable_pipeline_parallelism=True,
    pipeline_stages=4,
    pipeline_overlap_ratio=0.7,
    
    # Adaptive Batch Sizing Parameters
    enable_adaptive_batching=True,
    initial_batch_size=32,
    min_batch_size=8,
    max_batch_size=256,
    batch_adaptation_frequency=100,
    memory_threshold_high=0.85,
    memory_threshold_low=0.6,
    
    # Smart Data Sampling Parametersa
    enable_smart_sampling=True,
    sample_importance_weight=0.6,
    sample_diversity_weight=0.4,
    initial_sample_ratio=0.3,
    complexity_analysis_enabled=True,
    
    # Multi-input/output parameters
    num_inputs=1,
    num_outputs=1,
    output_dims=[64],  # Directly pass the list value
    
    # Self-supervised learning
    ssl_dim=128,
    ssl_weight=0.1,
    ssl_temperature=0.07,
    ssl_noise_std=0.1,
    
    # Spatiotemporal Processing
    use_spatial=True,
    
    # WINA Attention
    use_wina_attention=True,
    
    # Multi-task Learning Parameters
    max_tasks=50,
    diffusion_steps=1000,
    ctm_diffusion_coupling_strength=0.8,
    vocab_size=None,
    #enable_enhanced_mcmc=False, #ONLY USE THE ARC_AGI NOTEBOOK VERSION AND NOT THE ONE IMPORTED FROM THE DIFFUSION_NEWNEW file (This needs to be false). This flie cannot use this variable.
    #mcmc_config=MCMC_CONFIG_ARC, #I don't think this is needed. 
    output_audio_bytes=False
)

print("✓ EnhancedCTMConfig for ARC (config_arc_diffusion) created.")

if 'enhanced_ctm_mcmc' not in globals():
    print("Warning: 'enhanced_ctm_mcmc' not found in globals. Defaulting to None. Ensure the cell defining it (approx. lines 1820-1866) was run successfully.")
    enhanced_ctm_mcmc = None
    
if 'EnhancedCTMDiffusion' in globals() and EnhancedCTMDiffusion is not None:
    ctm_model_arc = EnhancedCTMDiffusion(config=config_arc_diffusion).to(device)
    print("✓ EnhancedCTMDiffusion model for ARC (ctm_model_arc) initialized.")

    # The external ARC output head will take features from the CTM core part of EnhancedCTMDiffusion
    arc_output_head_input_dim = config_arc_diffusion.output_dims[0]
    arc_output_head = nn.Linear(arc_output_head_input_dim, ARC_OUTPUT_HEAD_DIM).to(device)
    print(f"✓ ARC Output Head initialized (input_dim: {arc_output_head_input_dim}, output_dim: {ARC_OUTPUT_HEAD_DIM}).")

    # Handle external MCMC integration if enabled
    if 'enhanced_ctm_mcmc' in globals() and ENABLE_CTM_MCMC_INTEGRATION_FOR_ARC and enhanced_ctm_mcmc:
        # Ensure the external MCMC module's input_dim matches the new CTM's output
        if enhanced_ctm_mcmc.thought_network[0].in_features != config_arc_diffusion.output_dims[0]:
            print(f"Re-initializing external enhanced_ctm_mcmc for new input_dim {config_arc_diffusion.output_dims[0]}")
            # This part of the code assumes 'EnhancedCTMFenchelYoungIntegration' and other related variables are defined.
            # If not, this will raise an error, which is expected behavior if setup is wrong.
            enhanced_ctm_mcmc = EnhancedCTMFenchelYoungIntegration(
                input_dim=config_arc_diffusion.output_dims[0],
                output_space=arc_grid_output_space,
                mcmc_config=MCMC_CONFIG_ARC,
                use_large_neighborhood_search=True,
                lns_frequency=5,
                lns_neighborhood_size=10
            )
        ctm_mcmc_integration_arc = enhanced_ctm_mcmc.to(device) if enhanced_ctm_mcmc else None
        print(f"✓ External MCMC Integration for ARC is {'enabled' if ctm_mcmc_integration_arc else 'FAILED to enable'}.")
    else:
        ctm_mcmc_integration_arc = None

    arc_trainable_params = list(ctm_model_arc.parameters())
    if arc_output_head:
        arc_trainable_params.extend(list(arc_output_head.parameters()))
    if ctm_mcmc_integration_arc:
        arc_trainable_params.extend(list(ctm_mcmc_integration_arc.parameters()))

    optimizer_arc = optim.AdamW([p for p in arc_trainable_params if p.requires_grad], lr=LEARNING_RATE, weight_decay=1e-4)

    if ACCELERATE_AVAILABLE:
        print(" -> Preparing components with Hugging Face Accelerate...")
        accelerator_arc = Accelerator()
        components_to_prepare = [ctm_model_arc, optimizer_arc]
        if arc_output_head:
            components_to_prepare.insert(1, arc_output_head)
        if ctm_mcmc_integration_arc:
            components_to_prepare.insert(2, ctm_mcmc_integration_arc)
        
        prepared_components = accelerator_arc.prepare(*components_to_prepare)
        
        # Unpack the prepared components carefully
        ctm_model_arc = prepared_components[0]
        next_idx = 1
        if arc_output_head:
            arc_output_head = prepared_components[next_idx]
            next_idx += 1
        if ctm_mcmc_integration_arc:
            ctm_mcmc_integration_arc = prepared_components[next_idx]
            next_idx += 1
        optimizer_arc = prepared_components[next_idx]

        print("✓ ARC models and optimizer prepared with Accelerate.")
else:
    print("⚠️ Hugging Face Accelerate not available. Running on a single device.")

def find_directory(start_path, dir_name):
    """Recursively finds a directory by name."""
    for root, dirs, _ in os.walk(start_path):
        if dir_name in dirs:
            found_path = os.path.join(root, dir_name)
            print(f"Found '{dir_name}' directory at: {found_path}")
            return found_path
    return None

def find_file_directory(start_path, filename):
    """Recursively finds a file and returns its directory."""
    for root, _, files in os.walk(start_path):
        if filename in files:
            found_dir = os.path.abspath(root)
            print(f"Found '{filename}' in directory: {found_dir}")
            return found_dir
    print(f"Warning: File '{filename}' not found starting from '{start_path}'.")
    
print("\n--- Searching for evaluation and checkpoint directories ---")
# Path to ARC evaluation tasks
# Path to CTM checkpoints
CHECKPOINT_DIR_ARC_SEARCHED = find_directory(".", "ctm_arc_agi_2_enhanced_diffusion")

# --- Search for the specific evaluation file to determine the data directory ---
evaluation_filename = "16b78196.json"
dynamic_eval_dir = find_file_directory(".", evaluation_filename)

if dynamic_eval_dir:
    print(f"-> Evaluation directory dynamically set to: '{dynamic_eval_dir}'")
    ARC_EVAL_DIR = dynamic_eval_dir
else:
    print(f"-> Specific evaluation file '{evaluation_filename}' not found. Falling back to default path.")
    evaluation_relative_path = "contineous-thought-machines/data/evaluation"
    ARC_EVAL_DIR = os.path.abspath(evaluation_relative_path)
CHECKPOINT_DIR_ARC = CHECKPOINT_DIR_ARC_SEARCHED if CHECKPOINT_DIR_ARC_SEARCHED else os.path.join("checkpoints", "ctm_arc_agi_2_enhanced_diffusion")

if not CHECKPOINT_DIR_ARC_SEARCHED:
    print(f"-> Checkpoint directory not found dynamically, using fallback: '{CHECKPOINT_DIR_ARC}'")

NUM_EPOCHS_ARC = 20



class NewCustomARCGridDataset(Dataset):
    def __init__(self, data_path, max_grid_size=MAX_GRID_SIZE, padding_value=PADDING_VALUE):
        self.data_path = data_path
        self.task_files = []
        if os.path.isfile(data_path):
            if data_path.endswith(".json"):
                self.task_files.append(data_path)
        elif os.path.isdir(data_path):
            for root, _, files in os.walk(data_path):
                for file in files:
                    if file.endswith(".json"):
                        self.task_files.append(os.path.join(root, file))
        else:
            print(f"Error: Provided data path does not exist or is not a file/directory: {data_path}")
            self.tasks = []
            return

        if not self.task_files:
            print(f"Warning: No .json files found at path: {data_path}")
            self.tasks = []
            return
        
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

# --- Safetensors loading fix ---
# The load_file_safely function has been removed.
# Direct use of 'load_file' from safetensors.torch is now used in the main evaluation loop.

# --- Dataloader Initialization for Evaluation ---
print("\n--- Initializing Evaluation Dataloader ---")
arc_eval_loader = None
if 'ARC_EVAL_DIR' in globals() and os.path.exists(ARC_EVAL_DIR):
    print(f"  > Using NewCustomARCGridDataset for evaluation from path: {ARC_EVAL_DIR}")
    arc_eval_dataset = NewCustomARCGridDataset(
        data_path=ARC_EVAL_DIR,
        max_grid_size=MAX_GRID_SIZE,
        padding_value=PADDING_VALUE
    )
    if len(arc_eval_dataset) > 0:
        arc_eval_loader = DataLoader(arc_eval_dataset, batch_size=1, shuffle=False)
        print(f"✓ Evaluation DataLoader initialized with {len(arc_eval_dataset)} tasks.")
    else:
        print("⚠️ Evaluation dataset is empty. Skipping evaluation.")
else:
    print(f"⚠️ Evaluation directory not found or not specified (ARC_EVAL_DIR='{globals().get('ARC_EVAL_DIR', 'Not Set')}'). Cannot create DataLoader.")

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
            # Load state_dict using the safetensors library directly, as per user feedback
            state_dict_ctm = load_file(ctm_checkpoint_path_eval, device="cpu")
            ctm_model_arc.load_state_dict(state_dict_ctm, strict=False)
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
                    for trial in range(2):
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