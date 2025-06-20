ARC_EVAL_DIR = "/workspace/Arc-AGI-2/contineous-thought-machines/data/evaluation"  #Evaluation Dataset Directory.

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

ctm_model_arc, arc_output_head, optimizer_arc, ctm_mcmc_integration_arc, accelerator_arc = None, None, None, None, None

print("\n-----------------------------------------------------------------------------")
print("Initializing Configuration and Model for ARC with EnhancedCTMDiffusion")
print("-----------------------------------------------------------------------------")

'''
You do not need to add any of the variables again from the ctm_Diffusion_NEWNEW.py file to your config_arc_diffusion in the Arc_AGI_2_Final.ipynb file. All the parameters you listed are already explicitly defined when config_arc_diffusion is created (between lines 2200 and 2378 approximately).

The EnhancedCTMConfig class in ctm_Diffusion_NEWNEW.py provides default values for its fields. When you create an instance like config_arc_diffusion, any parameters you explicitly set will override these defaults. Since all the parameters in your list are already set in your notebook, those are the values that will be used for training.

For example:

attention_type is set to "subquadratic" on line 2260.
positional_embedding_type is set to 'multi-learnable-fourier' on line 2276.
enable_pipeline_parallelism is set to True on line 2288.
And so on for all the other parameters you mentioned.
If you wish to change any of these settings, you should modify their values directly in the existing config_arc_diffusion definition within your Arc_AGI_2_Final.ipynb file.
'''

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
    attention_type="subquadratic",
    subquadratic_attn_epsilon=1e-6,
    subquadratic_attn_poly_degree=5,
    attention_qkv_bias=True,
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
    use_spatial=True,
    use_wina_attention=True,
    max_tasks=50,
    diffusion_steps=1000,
    ctm_diffusion_coupling_strength=0.8,
    vocab_size=None,
    output_audio_bytes=False,
    unet_input_feature_dim=8192
)
print("✓ EnhancedCTMConfig for ARC (config_arc_diffusion) created.")

if 'enhanced_ctm_mcmc' not in globals():
    print("Warning: 'enhanced_ctm_mcmc' not found in globals. Defaulting to None. Ensure the cell defining it (approx. lines 1820-1866) was run successfully.")
    enhanced_ctm_mcmc = None
    
if 'EnhancedCTMDiffusion' in globals() and EnhancedCTMDiffusion is not None:
    ctm_model_arc = EnhancedCTMDiffusion(config=config_arc_diffusion).to(device)
    print("✓ EnhancedCTMDiffusion model for ARC (ctm_model_arc) initialized.")

    # The external ARC output head will take features from the CTM core part of EnhancedCTMDiffusion
    arc_output_head_input_dim = config_arc_diffusion.ctm_out_dims
    arc_output_head = nn.Linear(arc_output_head_input_dim, ARC_OUTPUT_HEAD_DIM).to(device)
    print(f"✓ ARC Output Head initialized (input_dim: {arc_output_head_input_dim}, output_dim: {ARC_OUTPUT_HEAD_DIM}).")

    # Handle external MCMC integration if enabled
    if ENABLE_CTM_MCMC_INTEGRATION_FOR_ARC and enhanced_ctm_mcmc:
        # Ensure the external MCMC module's input_dim matches the new CTM's output
        if enhanced_ctm_mcmc.thought_network[0].in_features != config_arc_diffusion.ctm_out_dims:
            print(f"Re-initializing external enhanced_ctm_mcmc for new input_dim {config_arc_diffusion.ctm_out_dims}")
            enhanced_ctm_mcmc = EnhancedCTMFenchelYoungIntegration(
                input_dim=config_arc_diffusion.ctm_out_dims, # Use output dim of CTM core
                output_space=arc_grid_output_space,
                mcmc_config=MCMC_CONFIG_ARC,
                use_large_neighborhood_search=True,
                lns_frequency=5,
                lns_neighborhood_size=10
            )
        ctm_mcmc_integration_arc = enhanced_ctm_mcmc.to(device) if enhanced_ctm_mcmc else None
        print(f"✓ External MCMC Integration for ARC is {'enabled' if ctm_mcmc_integration_arc else 'FAILED to enable'}.")
    
    arc_trainable_params = list(ctm_model_arc.parameters()) # EnhancedCTMDiffusion parameters
    if arc_output_head: arc_trainable_params.extend(list(arc_output_head.parameters()))
    if ctm_mcmc_integration_arc:
        arc_trainable_params.extend(list(ctm_mcmc_integration_arc.parameters()))

    optimizer_arc = optim.AdamW([p for p in arc_trainable_params if p.requires_grad], lr=LEARNING_RATE, weight_decay=1e-4)
    
    if ACCELERATE_AVAILABLE:
        accelerator_arc = Accelerator()
        models_to_prepare = [ctm_model_arc] # Start with the main model
        if arc_output_head: models_to_prepare.append(arc_output_head)
        if ctm_mcmc_integration_arc: models_to_prepare.append(ctm_mcmc_integration_arc)
        
        prepared_components = accelerator_arc.prepare(*models_to_prepare, optimizer_arc)
        
        optimizer_arc = prepared_components[-1] # Last element is the optimizer
        prepared_models_tuple = prepared_components[:-1] # All other elements are models

        ctm_model_arc = prepared_models_tuple[0]
        model_idx = 1
        if arc_output_head:
            arc_output_head = prepared_models_tuple[model_idx]
            model_idx +=1
        if ctm_mcmc_integration_arc:
            ctm_mcmc_integration_arc = prepared_models_tuple[model_idx]
        print("✓ ARC models (EnhancedCTMDiffusion) and optimizer prepared with Accelerate.")
else:
    print("⚠️ EnhancedCTMDiffusion model or its config for ARC-AGI-2 could not be initialized. Check imports.")

CHECKPOINT_DIR_ARC = os.path.join(CHECKPOINT_DIR, "ctm_arc_agi_2_enhanced_diffusion") # New checkpoint dir
os.makedirs(CHECKPOINT_DIR_ARC, exist_ok=True)
print(f"ARC Checkpoints will be saved to: {CHECKPOINT_DIR_ARC}")

NUM_EPOCHS_ARC = 20
ARC_BATCH_SIZE = 8

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

arc_criterion = nn.CrossEntropyLoss(ignore_index=PADDING_VALUE)
print("\n✓ ARC-AGI-2 Setup Complete.")