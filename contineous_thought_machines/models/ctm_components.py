"""
Shared components for CTM models.

This file contains shared components for the Continuous Thought Machine (CTM)
models, moved from ctm_HRM.py to break circular dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
import math
import numpy as np
import time
import pickle
import os
import hashlib
import threading
import random
import copy
from concurrent.futures import ThreadPoolExecutor
import queue

from diffusers import DPMSolverMultistepScheduler
from torch.nn import GRU

# Local imports
from .modules import SynapseUNET, SuperLinear, Squeeze, LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding
from .utils import compute_normalized_entropy, TaskAnalyzer
from .long_term_memory import LongTermMemory, MemoryReplayPolicy
from .mamba_block import Mamba2Block
from .enhanced_neuron_selection import EnhancedNeuronSelector
from .biological_neuron_selection import BiologicalNeuronSelector, BiologicalSelectionConfig
from .constants import VALID_NEURON_SELECT_TYPES, VALID_POSITIONAL_EMBEDDING_TYPES
import torch.quantization as quant
from .mamba_block import quantize_adaptive, dequantize_adaptive, BitwidthAdapter, QuantizationPolicyNetwork, SelectiveQuantizer, load_quantized_state, dequantize_for_adaptation, quantize_after_adaptation

from .neuromodulators import (
    BaseNeuromodulator,
    DopamineModulator,
    SerotoninModulator,
    OxytocinModulator,
    NorepinephrineModulator,
    AcetylcholineModulator,
    EndorphinsModulator,
    CortisolModulator,
    GABAModulator,
    GlutamateModulator
)
try:
    import sys
    sys.path.append("/workspaces/Arc-AGI-2/contineous_thought_machines/models/flash-attention-3")
    from flash_attn import flash_attention
except ImportError:
    flash_attention = None

@dataclass
class EnhancedCTMConfig: # Renamed from ContinualLearningConfig for consistency in the target file
    """Enhanced configuration for continual learning CTM-diffusion model,
    incorporating binary processing, multi-task learning, and advanced CTM features."""
    
    # Model architecture (General Transformer/Diffusion settings)
    d_model: int = 512  # Main model dimensionality
    n_heads: int = 8
    n_layers: int = 24
    max_sequence_length: int = 8192 # Max input sequence length in terms of bytes or patches #This is the old original size before changes: 8192.
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

    # Neuromodulator Configuration
    enable_neuromodulators: bool = True
    neuromodulator_dim: int = 512  # Dimension for neuromodulator computations
    active_neuromodulators: List[str] = field(default_factory=lambda: [
        'dopamine', 'serotonin', 'oxytocin', 'norepinephrine', 'acetylcholine',
        'endorphins', 'cortisol', 'gaba', 'glutamate'
    ])
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
    ctm_use_qat: bool = True #Turns quant-aware training off and on.
    ctm_adaptive_quantization: bool = True
    ctm_quant_min_bits: int = 2
    ctm_quant_max_bits: int = 8
    ctm_quant_policy_search: bool = True
    ctm_selective_quantization: bool = True
    quant_enabled_training: bool = False
    quant_enabled_inference: bool = False
    
    # Diffusion Parameters
    diffusion_steps: int = 1000
    noise_schedule: str = "cosine" # e.g., "linear", "cosine"
    diffusion_beta_start: float = 0.0001
    diffusion_beta_end: float = 0.02
    diffusion_timesteps: int = 1000 # Number of timesteps for the diffusion process
    ctm_diffusion_coupling_strength: float = 0.8 # How CTM influences diffusion
    adaptive_scheduling: bool = True  # CTM-adaptive diffusion timestep scheduling
    iterative_refinement: bool = True # Iterative CTM-diffusion refinement for sampling
    
    #Inferred_Latent_Dimensions Set to Avoid runtime errors but it does not functionally do anything in the model or program processing. 
    inferred_task_latent_dim=512
    
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
    attention_type: str = "WINA"  # Options: "standard", "binary_sparse", "WINA" #subquadratic was removed and the primary attention will now be WINA.
    control_dim: int = 64  # for MetaWINASparsifier

    # Positional Embedding Parameters
    positional_embedding_type: Optional[str] = 'multi-learnable-fourier' # e.g., 'custom-rotational-1d', 'learnable-fourier', multi-learnable-fourier' #Can set the value here. 
    positional_embedding_dim: Optional[int] = None  # Dimension of the positional embedding, defaults to ctm_input_dim if None
    reshape_patch_sequence_to_grid: bool = True # If True, reshape patch sequence to a 2D grid for 2D PEs. Must set to true if using 2D Grid for Positional Embeddings.
    patch_grid_width: Optional[int] = None       # Desired width of the patch grid if reshaping

    # --- Hierarchical Reasoning Model (HRM) Parameters ---
    use_hrm_core: bool = True # Set to True to use the HierarchicalCTM core
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
    use_spatial: bool = False  # Enable spatial processing for image/video data in the ctm model (does not affect Diffusion Model)
    
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

    # --- Consciousness Controller Parameters ---
    enable_consciousness_controller: bool = True
    consciousness_max_attention_steps: int = 100

    # --- Synaptic Empathy Parameters ---
    enable_synaptic_empathy: bool = True # Set to True to use the new SynapticEmpathy module
    synaptic_empathy_reward_weight: float = 0.1

    # --- Mirror Neuron / High-Level Empathy Parameters ---
    enable_mirror_neurons: bool = True # Set to True to use the high-level MirrorNeuronLayer
    num_emotion_dim: int = 4 # Dimensionality of the emotion state vector
    goal_dim: int = 8 # Dimensionality of the predicted goal vector
    mirror_reward_weight: float = 0.2 # Weight for the selfless reward signal

  # --- Recursion Parameters ---
    max_recursion: int = 3
    early_stop_threshold: float = 1e-3
    
# --- Confidence Thresholding Parameters ---
    confidence_threshold: float = 0.0 # Confidence threshold for abstaining. If > 0, model can abstain.

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
           VALID_NEURON_SELECT_TYPES is not None and self.ctm_neuron_select_type not in VALID_NEURON_SELECT_TYPES:
            print(f"Warning: ctm_neuron_select_type '{self.ctm_neuron_select_type}' is not in VALID_NEuron_SELECT_TYPES ({VALID_NEURON_SELECT_TYPES}).")

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

        # Validate quantization params
        if self.ctm_use_qat:
            if self.ctm_quant_min_bits >= self.ctm_quant_max_bits:
                raise ValueError("ctm_quant_min_bits must be < ctm_quant_max_bits")
            if self.ctm_adaptive_quantization and not self.ctm_quant_policy_search:
                print("Warning: Adaptive quantization enabled without policy search.")

class WorkingMemoryBuffer(nn.Module):
    def __init__(self, d_model, capacity=5):
        super().__init__()
        self.capacity = capacity
        self.buffer = None
        self.proj = nn.Linear(d_model, d_model)

    def update(self, item):
        if self.buffer is None:
            self.buffer = item.unsqueeze(1)
        else:
            self.buffer = torch.cat([self.buffer, item.unsqueeze(1)], dim=1)[:, -self.capacity:]
        return self.proj(self.buffer.mean(dim=1))

class NeuromodulatorManager(nn.Module):
    def __init__(self, config: EnhancedCTMConfig):
        super().__init__()
        self.neuromodulators = nn.ModuleDict()
        for mod_name in config.active_neuromodulators:
            mod_class = globals()[f"{mod_name.capitalize()}Modulator"]
            self.neuromodulators[mod_name] = mod_class(config.neuromodulator_dim)
        self.fusion_layer = nn.Linear(len(config.active_neuromodulators) * config.neuromodulator_dim, config.neuromodulator_dim)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        outputs = []
        for mod in self.neuromodulators.values():
            outputs.append(mod(input_tensor))
        concatenated = torch.cat(outputs, dim=-1)
        fused = self.fusion_layer(concatenated)
        return fused
    
class HRM_H_Module(nn.Module):
    """The High-Level, slow-updating recurrent module for the HR-CTM."""
    def __init__(self, config: EnhancedCTMConfig):
        super().__init__()
        self.config = config
        self.base_thresholds = {'critical': 0.99, 'medium': 0.8, 'low': 0.5}
        self.confidence_thresholds = {k: 0.0 for k in self.base_thresholds}  # Start at 0
        self.initial_epochs = 5  # Epochs before ramp-up starts
        self.current_epoch = 0
        # This module integrates the result from the L-module (zL) into its own state (zH).
        self.mamba = Mamba2Block(d_model=config.d_model)
        self.sparse_attn = WINAAttention(d_model=config.d_model, n_heads=config.n_heads, dropout=config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.planning_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.ReLU(),
            nn.Linear(config.d_model * 2, config.d_model * 4),
            nn.ReLU(),
            nn.Linear(config.d_model * 4, config.d_model * 2),
            nn.ReLU(),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model)
        )
        self.norm3 = nn.LayerNorm(config.d_model)
        # Project zL to match d_model for attention
        self.zl_proj = nn.Linear(config.d_model, config.d_model)  # Assuming zL has d_model
        patcher_config = {
            'embedding_dim': config.patch_embedding_dim,
            'patch_cnn_channels': config.patch_encoder_cnn_channels,
            'patching_mode': config.entropy_patcher_threshold_type,
            'global_threshold': config.entropy_patcher_global_threshold,
            'relative_threshold': config.entropy_patcher_relative_threshold,
            'min_patch_size': config.entropy_patcher_min_patch_size,
            'max_patch_size': config.entropy_patcher_max_patch_size,
            'entropy_byte_vocab_size': config.entropy_model_byte_vocab_size,
            'entropy_embedding_dim': config.entropy_model_embedding_dim,
            'entropy_hidden_dim': config.entropy_model_hidden_dim,
            'entropy_num_layers': config.entropy_model_num_layers,
            'entropy_dropout': config.entropy_model_dropout
        }
        self.program_synthesizer = ProgramSynthesizer(
            d_model=config.d_model,
            n_heads=config.program_synth_n_heads,
            n_layers=config.program_synth_n_layers,
            d_ff=config.program_synth_d_ff,
            dropout=config.dropout,
            max_gen_len=config.max_sequence_length, # Or a more specific config
            patcher_config=patcher_config
        )
        self.hypernet = HyperNetwork(config.d_model * 2, config.d_model)
        self.meta_learner = nn.Linear(config.d_model * 2, config.d_model)  # Base learner, params generated by hypernet
        self.foresight = ForesightSimulator(config.d_model)
        self.max_recursion = config.max_recursion
        self.early_stop_threshold = config.early_stop_threshold
        self.program_feedback_proj = nn.Linear(config.d_model, config.d_model)
        self.thought_ctm = OriginalCTMCore(config)
        self.thought_feedback_proj = nn.Linear(config.ctm_out_dims, config.d_model)
        
        # Add CTM-like components for H-module
        self.h_synapses = SuperLinear(config.d_model * 2, config.d_model, depth=config.ctm_synapse_depth, dropout=config.ctm_dropout)
        self.h_trace_processor = SuperLinear(config.d_model * config.ctm_memory_length, config.d_model, depth=config.ctm_deep_nlms, dropout=config.ctm_dropout)
        self.h_q_proj = nn.Linear(config.d_model, config.d_model)  # For H-module sync-based query

    def update_thresholds(self, epoch, total_epochs):
        self.current_epoch = epoch
        if epoch < self.initial_epochs:
            factor = 0.0
        else:
            factor = (epoch - self.initial_epochs) / max(1, total_epochs - self.initial_epochs)
        
        for level in self.confidence_thresholds:
            self.confidence_thresholds[level] = factor * self.base_thresholds[level]

    def forward(self, zH: torch.Tensor, zL: torch.Tensor, retrieved_memory: torch.Tensor, thought_guidance: bool = True, confidence_level: str = 'medium') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            zH (torch.Tensor): Current high-level state.
            zL (torch.Tensor): Final low-level state from the L-cycle.
            retrieved_memory (torch.Tensor): Memory retrieved from the LTM.
            thought_guidance (bool): Flag to switch to direct CTM thought vector guidance. #Recommended on for most model usage.
            confidence_level (str): 'critical', 'medium', or 'low'
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Next high-level state, encoded_patches (or None), patch_indices (or None), entropy_aux_loss (or 0).
        """
        current_zH = zH
        prev_zH = None
        depth = 0
        encoded_patches = None
        patch_indices = None
        entropy_aux_loss = torch.tensor(0.0, device=zH.device)
        deltas = []
        
        # Initialize H-module trace
        h_trace = torch.zeros_like(current_zH.unsqueeze(-1).repeat(1, 1, self.config.ctm_memory_length))
        
        # Initialize sync for H-module
        decay_alpha_h, decay_beta_h = None, None
        r_h = torch.exp(torch.tensor(-0.1))  # Example decay rate
        
        while depth < self.max_recursion:
            # Compute H-module synchronization (pulsing)
            sync_h, decay_alpha_h, decay_beta_h = self.compute_synchronisation(
                current_zH, decay_alpha_h, decay_beta_h, r_h, 'action'  # Reuse 'action' type
            )
            
            # The query is the current high-level state modulated by sync
            q = self.h_q_proj(sync_h).unsqueeze(1)
            
            # The key/value is the information from the completed low-level cycle and retrieved memory
            # The retrieved_memory is now a single contextualized vector from the LTM's attention mechanism
            kv = self.zl_proj(zL) + retrieved_memory.squeeze(0) # Squeeze to remove the batch dim of 1
            # Attention step
            kv = (self.zl_proj(zL) + retrieved_memory.squeeze(0)).unsqueeze(1)
            current_zH = current_zH.unsqueeze(1)
            q = self.h_q_proj(sync_h).unsqueeze(1)
            
            # Dynamic routing: Compute WINA scores to decide per token
            scores = self.sparse_attn.wina_sparsifier.compute_wina_scores(current_zH, self.sparse_attn.q_proj.weight)
            route_to_attention = (scores > 0.5).float()  # Example threshold; make learnable
            
            # Mamba path (default/efficient)
            mamba_out = self.mamba(current_zH, confidence_level=confidence_level)
            
            # Sparse attention path (selective)
            attn_out = self.sparse_attn(current_zH, kv, kv)
            
            # Fuse based on routing
            current_zH = current_zH + (mamba_out * (1 - route_to_attention) + attn_out * route_to_attention)
            
            # Repeat for second block (or loop for more)
            scores = self.sparse_attn.wina_sparsifier.compute_wina_scores(current_zH, self.sparse_attn.q_proj.weight)
            route_to_attention = (scores > 0.5).float()
            
            mamba_out = self.mamba(current_zH, confidence_level=confidence_level)
            attn_out = self.sparse_attn(current_zH, kv, kv)
            current_zH = current_zH + (mamba_out * (1 - route_to_attention) + attn_out * route_to_attention)
            
            current_zH = current_zH.squeeze(1)
            
            meta_input = torch.cat([current_zH, zL], dim=-1)
            # Dynamic meta-learning with hypernetwork
            weight, bias = self.hypernet(meta_input)
            meta_update = F.linear(meta_input, weight, bias)
            current_zH = current_zH + meta_update * 0.1  # Small meta-update step
            current_zH = self.norm1(current_zH)
            
            # Additional planning layer
            planning_output = self.planning_mlp(current_zH)
            current_zH = self.norm3(current_zH + planning_output)
            
            # Add foresight simulation
            foresight_adjust = self.foresight(current_zH)
            current_zH = current_zH + foresight_adjust * 0.05
            
            # Add CTM-like synapse and NLM processing
            h_pre_synapse = torch.cat([current_zH, retrieved_memory.squeeze(0)], dim=-1)
            h_state = self.h_synapses(h_pre_synapse)
            h_trace = torch.cat((h_trace[:, :, 1:], h_state.unsqueeze(-1)), dim=-1)
            current_zH = self.h_trace_processor(h_trace.view(h_trace.shape[0], -1))
            
            if not thought_guidance:
                # Synthesize a program using the new synthesizer
                encoded_patches, patch_indices, entropy_aux_loss = self.program_synthesizer(current_zH)
                
                # Feedback from synthesized program to high-level state
                if encoded_patches is not None and encoded_patches.size(1) > 0:
                    program_feedback = self.program_feedback_proj(encoded_patches.mean(dim=1))
                    current_zH = current_zH + program_feedback * 0.1
            else:
                # Direct CTM thought vector guidance
                ctm_predictions, ctm_certainties, ctm_sync_out = self.thought_ctm(current_zH.unsqueeze(1))
                thought_feedback = self.thought_feedback_proj(ctm_sync_out)
                current_zH = current_zH + thought_feedback * 0.1
                # Set placeholders for return values
                encoded_patches = None
                patch_indices = None
                entropy_aux_loss = torch.tensor(0.0, device=zH.device)
            
            # Early stopping check
            if prev_zH is not None:
                delta = torch.norm(current_zH - prev_zH, dim=-1).mean()
                deltas.append(delta)
                if delta < self.early_stop_threshold:
                    break
            
            prev_zH = current_zH.clone()
            depth += 1
        
        # Hallucination reduction: Compute confidence based on variance of deltas
        if deltas:
            variance = torch.var(torch.stack(deltas))
            confidence = torch.exp(-variance)
            threshold = self.confidence_thresholds.get(confidence_level, 0.8)
            if not self.training and confidence < threshold:
                current_zH = current_zH * 0  # Abstain only during inference
        else:
            confidence = torch.tensor(1.0, device=zH.device)
        
        # The 'program' is now the sequence of encoded patches (or None in direct mode).
        # The other outputs might be used for loss calculation or debugging.
        return current_zH, encoded_patches, patch_indices, entropy_aux_loss, confidence

class HRM_L_Module(nn.Module):
    """The Low-Level, fast-updating CTM-based recurrent module for the HR-CTM."""
    def __init__(self, config: EnhancedCTMConfig, parent_ctm: 'HierarchicalCTM'):
        super().__init__()
        self.config = config
        self.d_model = config.ctm_d_model
        self.d_input = config.ctm_input_dim
        
        self.mamba_encoder = Mamba2Block(d_model=self.d_input)
        
        # Inherit synapse and NLM models from parent HierarchicalCTM
        # to ensure they are registered correctly under the main model.
        self.synapses = parent_ctm.synapses
        self.trace_processor = parent_ctm.trace_processor
        
        # Projector for the query, derived from the low-level sync state
        self.q_proj = nn.Linear(parent_ctm.synch_representation_size_action, self.d_input)
        self.top_down_projector = nn.Linear(self.config.d_model, self.d_model)  # Project zH to modulation signal
        
        if self.config.use_spatial:
            self.spatial_reasoning = SpatialReasoningModule(self.d_model)
            self.three_d_spatial_reasoning = ThreeDSpatialReasoningModule(self.d_model)
        else:
            self.spatial_reasoning = None
            self.three_d_spatial_reasoning = None

    def forward(self,
                activated_zL: torch.Tensor,
                zL_trace: torch.Tensor,
                zH: torch.Tensor,
                x_context: torch.Tensor,
                sync_action: torch.Tensor,
                confidence_level: str = 'medium') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one step of the low-level CTM computation.
    
        Args:
            activated_zL: Current post-activation state of the L-module. (B, D)
            zL_trace: History of pre-activations for the L-module. (B, D, M)
            zH: Current high-level state, provides top-down context. (B, D)
            x_context: External input, provides bottom-up context. (B, S, d_input)
            sync_action: Synchronization representation for generating attention query. (B, sync_dim)
    
        Returns:
            A tuple of (next_activated_zL, next_zL_trace).
        """
        x_context = self.mamba_encoder(x_context, confidence_level=confidence_level)
    
        # 1. Interact with context via attention
        # Query is from L-module's own action synchronisation
        q = self.q_proj(sync_action).unsqueeze(1)
        
        # Key/Value is a combination of external input and high-level context
        # Add zH as part of the key/value context
        kv = x_context + zH.unsqueeze(1)
        attn_out, _ = self.attention(q, kv, kv)
        attn_out = attn_out.squeeze(1)
    
        # 2. Form input for synapses
        pre_synapse_input = torch.cat((attn_out, activated_zL), dim=-1)
    
        # 3. Apply Synapses to get pre-activation state
        state = self.synapses(pre_synapse_input)
        top_down_mod = self.top_down_projector(zH)  # (B, D)
        state = state + top_down_mod * 0.3  # Modulate with strength 0.3
        
        # Add parietal-inspired spatial reasoning
        if self.config.use_spatial and self.spatial_reasoning is not None:
            state = self.spatial_reasoning(state.unsqueeze(1)).squeeze(1)
    
        # Add 3D spatial reasoning - assume a 3D grid size, e.g., (4,4,4) if d_model=64
        # Adjust based on actual d_model; here assuming d_model is cube-able
        if self.config.use_spatial and self.three_d_spatial_reasoning is not None:
            cube_root = int(self.d_model ** (1/3))
            grid_3d = (cube_root, cube_root, cube_root)
            state = self.three_d_spatial_reasoning(state.unsqueeze(1), grid_size=grid_3d).squeeze(1)
    
        # 4. Update state trace (memory for NLMs)
        next_zL_trace = torch.cat((zL_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
    
        # 5. Apply Neuron-Level Models (NLMs) to get next post-activation state
        next_activated_zL = self.trace_processor(next_zL_trace)
        
        return next_activated_zL, next_zL_trace

def batched_bytes_to_numeric_tensor(byte_batch_tensor: torch.Tensor, item_size: int = 4, target_dtype: np.dtype = np.dtype(np.float32)) -> torch.Tensor:
    """
    Converts a batch of byte tensors (uint8) to a batch of numeric tensors (e.g., float32).
    Assumes each row in byte_batch_tensor represents a sequence of bytes that can be
    interpreted as target_dtype, and its length is a multiple of item_size.
    """
    if byte_batch_tensor.ndim == 1: # Single sequence
        byte_batch_tensor = byte_batch_tensor.unsqueeze(0)

    if byte_batch_tensor.shape[-1] % item_size != 0:
        raise ValueError(f"Number of bytes ({byte_batch_tensor.shape[-1]}) must be divisible by item_size ({item_size}) for {target_dtype} conversion.")
    
    processed_list = []
    for i in range(byte_batch_tensor.shape[0]):
        # .cpu().numpy() is essential as frombuffer works on CPU byte buffers
        single_byte_seq_np = byte_batch_tensor[i].cpu().numpy()
        # Ensure the numpy array is C-contiguous for tobytes()
        if not single_byte_seq_np.flags['C_CONTIGUOUS']:
            single_byte_seq_np = np.ascontiguousarray(single_byte_seq_np)
        
        # Convert the uint8 numpy array to a byte string, then interpret as target_dtype
        np_numeric_values = np.frombuffer(single_byte_seq_np.tobytes(), dtype=target_dtype)
        # .copy() here ensures that the tensor owns its memory
        processed_list.append(torch.from_numpy(np_numeric_values.copy()))
        
    stacked_numeric_tensor = torch.stack(processed_list, dim=0).to(byte_batch_tensor.device)
    return stacked_numeric_tensor

def batched_numeric_tensor_to_bytes(numeric_batch_tensor: torch.Tensor, source_dtype: np.dtype = np.dtype(np.float32)) -> torch.Tensor:
    """
    Converts a batch of numeric tensors (e.g., float32) to a batch of byte tensors (uint8).
    """
    if numeric_batch_tensor.ndim == 1: # Single sequence
        numeric_batch_tensor = numeric_batch_tensor.unsqueeze(0)

    processed_list = []
    for i in range(numeric_batch_tensor.shape[0]):
        # .detach().cpu().numpy() and ensure it's the correct source_dtype before tobytes()
        single_numeric_seq_np = numeric_batch_tensor[i].detach().cpu().numpy().astype(source_dtype)
        # Ensure the numpy array is C-contiguous for tobytes()
        if not single_numeric_seq_np.flags['C_CONTIGUOUS']:
            single_numeric_seq_np = np.ascontiguousarray(single_numeric_seq_np)
            
        byte_data = single_numeric_seq_np.tobytes()
        # Convert the byte string back to a uint8 numpy array
        np_bytes = np.frombuffer(byte_data, dtype=np.uint8)
        # .copy() here ensures that the tensor owns its memory
        processed_list.append(torch.from_numpy(np_bytes.copy()))
        
    stacked_bytes = torch.stack(processed_list, dim=0).to(numeric_batch_tensor.device)
    return stacked_bytes

class GroupEquivariantAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, x, grid_size=None):
        if grid_size is None:
            return self.attn(x, x, x)[0]
        h, w = grid_size
        seq_len = x.shape[1]
        if h * w != seq_len:
            return self.attn(x, x, x)[0]
        outputs = []
        for rot in range(4):
            x_rot = self.rotate_tensor(x, rot, h, w)
            attn_out = self.attn(x_rot, x_rot, x_rot)[0]
            attn_out_back = self.rotate_tensor(attn_out, 4 - rot, h, w)
            outputs.append(attn_out_back)
        return torch.mean(torch.stack(outputs), dim=0)

    def rotate_tensor(self, tensor, k, h, w):
        grid = tensor.reshape(tensor.shape[0], h, w, tensor.shape[-1])
        rotated = torch.rot90(grid, k, [1, 2])
        return rotated.reshape(tensor.shape[0], h * w, tensor.shape[-1])

class SpatialReasoningModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.spatial_attn = GroupEquivariantAttention(d_model, num_heads=8)
        self.spatial_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, grid_size=None):
        attn_out = self.spatial_attn(x, grid_size=grid_size)
        return self.norm(x + self.spatial_mlp(attn_out))

class ThreeDEquivariantAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, x, grid_size=None):
        if grid_size is None:
            return self.attn(x, x, x)[0]
        h, w, d = grid_size
        seq_len = x.shape[1]
        if h * w * d != seq_len:
            return self.attn(x, x, x)[0]
        outputs = []
        # Apply rotations along different axes
        for axis in [0, 1, 2]:  # x, y, z axes
            for rot in [1, 2, 3]:  # 90, 180, 270 degrees
                x_rot = self.rotate_3d_tensor(x, axis, rot, h, w, d)
                attn_out = self.attn(x_rot, x_rot, x_rot)[0]
                # Rotate back
                attn_out_back = self.rotate_3d_tensor(attn_out, axis, 4 - rot, h, w, d)
                outputs.append(attn_out_back)
        return torch.mean(torch.stack(outputs), dim=0)

    def rotate_3d_tensor(self, tensor, axis, k, h, w, d):
        dims = [h, w, d]
        grid = tensor.reshape(tensor.shape[0], *dims, tensor.shape[-1])
        # Rotate along the specified axis
        if axis == 0:  # Rotate in yz-plane
            rotated = torch.rot90(grid, k, [2, 3])
        elif axis == 1:  # Rotate in xz-plane
            rotated = torch.rot90(grid, k, [1, 3])
        elif axis == 2:  # Rotate in xy-plane
            rotated = torch.rot90(grid, k, [1, 2])
        return rotated.reshape(tensor.shape[0], h * w * d, tensor.shape[-1])

class ThreeDSpatialReasoningModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.spatial_attn = ThreeDEquivariantAttention(d_model, num_heads=8)
        self.spatial_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, grid_size=None):
        attn_out = self.spatial_attn(x, grid_size=grid_size)
        return self.norm(x + self.spatial_mlp(attn_out))
            
class BinarySparseAttention(nn.Module):
    """
    Sparse attention mechanism optimized for binary data processing.
    Uses learned sparsity patterns and binary-aware attention computation.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, sparsity_ratio: float = 0.1,
                 binary_pattern_size: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sparsity_ratio = sparsity_ratio
        self.binary_pattern_size = binary_pattern_size
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Standard attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Binary pattern detection
        self.binary_pattern_detector = nn.Conv1d(
            embed_dim, num_heads, kernel_size=binary_pattern_size,
            padding=binary_pattern_size//2, groups=num_heads
        )
        
        # Learned sparsity mask generator
        self.sparsity_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def generate_sparse_mask(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Generate sparse attention mask based on binary patterns and learned importance."""
        batch_size, seq_len, _ = q.shape
        
        # Detect binary patterns
        q_patterns = self.binary_pattern_detector(q.transpose(1, 2)).transpose(1, 2)
        k_patterns = self.binary_pattern_detector(k.transpose(1, 2)).transpose(1, 2)
        
        # Compute pattern similarity
        pattern_sim = torch.bmm(q_patterns, k_patterns.transpose(1, 2))
        
        # Generate importance scores
        q_importance = self.sparsity_predictor(q).squeeze(-1)  # (batch, seq_len)
        k_importance = self.sparsity_predictor(k).squeeze(-1)  # (batch, seq_len)
        
        # Combine pattern similarity and importance
        importance_matrix = q_importance.unsqueeze(2) * k_importance.unsqueeze(1)
        combined_scores = pattern_sim.mean(dim=1) + importance_matrix
        
        # Create sparse mask
        num_keep = max(1, int(seq_len * self.sparsity_ratio))
        _, top_indices = torch.topk(combined_scores, num_keep, dim=-1)
        
        sparse_mask = torch.zeros_like(combined_scores, dtype=torch.bool)
        sparse_mask.scatter_(-1, top_indices, True)
        
        return sparse_mask
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None, need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with sparse binary-aware attention."""
        batch_size, seq_len, embed_dim = query.shape
        
        # Project to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Generate sparse mask
        sparse_mask = self.generate_sparse_mask(query, key)
        sparse_mask = sparse_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply sparse mask
        attn_scores = attn_scores.masked_fill(~sparse_mask, float('-inf'))
        
        # Apply additional attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_proj(attn_output)
        
        if need_weights:
            return output, attn_weights.mean(dim=1)  # Average over heads
        return output, None 
    
class WINASparsifier:
    """
    WINA (Weight Informed Neuron Activation) sparse activation framework.
    
    This implements the training-free sparse activation method that jointly considers
    hidden state magnitudes and column-wise ℓ2-norms of weight matrices for better
    approximation error bounds compared to magnitude-only methods like TEAL.
    
    Key improvements over existing methods:
    - Considers both input magnitudes AND weight importance
    - Provides theoretical guarantees for tighter approximation error bounds
    - Training-free and plug-and-play design
    - Supports heterogeneous sparsity across layers
    """
    
    def __init__(self, sparsity_ratio: float = 0.5, use_layer_specific: bool = True):
        """
        Initialize WINA sparsifier.
        
        Args:
            sparsity_ratio: Target sparsity ratio (0.0 to 1.0)
            use_layer_specific: Whether to use layer-specific sparsity ratios
        """
        self.sparsity_ratio = sparsity_ratio
        self.use_layer_specific = use_layer_specific
        self.layer_sparsity_ratios = {}
        
        # Cache for column norms to avoid recomputation
        self._column_norm_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def compute_column_norms(self, weight_matrix: torch.Tensor, cache_key: str = None) -> torch.Tensor:
        """
        Compute column-wise ℓ2-norms of weight matrix with caching.
        
        Args:
            weight_matrix: Weight matrix of shape [out_features, in_features]
            cache_key: Optional cache key for storing computed norms
            
        Returns:
            Column-wise ℓ2-norms of shape [in_features]
        """
        if cache_key and cache_key in self._column_norm_cache:
            self._cache_hits += 1
            return self._column_norm_cache[cache_key]
        
        self._cache_misses += 1
        column_norms = torch.norm(weight_matrix, dim=0, p=2)
        
        if cache_key:
            self._column_norm_cache[cache_key] = column_norms
            
        return column_norms
    
    def apply_wina_gating(self,
                         hidden_states: torch.Tensor,
                         weight_matrix: torch.Tensor,
                         layer_name: str = "default",
                         cache_key: str = None,
                         dynamic_sparsity: Optional[float] = None,
                         return_scores: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply WINA gating mechanism to hidden states.
        
        The WINA criterion selects neurons based on |x_i * c_i| where:
        - x_i is the hidden state magnitude
        - c_i is the column-wise ℓ2-norm of the weight matrix
        
        This joint consideration provides better approximation error bounds
        than methods that only use hidden state magnitudes.
        
        Args:
            hidden_states: Input tensor of shape [..., in_features]
            weight_matrix: Weight matrix of shape [out_features, in_features]
            layer_name: Name of the layer for layer-specific sparsity
            cache_key: Optional cache key for weight matrix norms
            dynamic_sparsity: Optional dynamic sparsity ratio to override the fixed one
            
        Returns:
            Gated hidden states with same shape as input
        """
        # Get the input dimension
        in_features = hidden_states.shape[-1]
        
        # Compute column-wise ℓ2-norms of weight matrix
        column_norms = self.compute_column_norms(weight_matrix, cache_key)
        
        # Ensure column_norms has the right shape for broadcasting
        if len(hidden_states.shape) > 2:
            # Handle batch dimensions by expanding column_norms appropriately
            expand_dims = [1] * (len(hidden_states.shape) - 1) + [in_features]
            column_norms = column_norms.view(*expand_dims)
        
        # Compute surprise: deviation from mean hidden state
        mean_hs = hidden_states.mean(dim=-1, keepdim=True)
        surprise = torch.abs(hidden_states - mean_hs)
        surprise = surprise / surprise.max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        
        # Compute WINA criterion: |x_i * c_i| * (1 + surprise_i) # where c_i is column norm
        wina_scores = torch.abs(hidden_states) * column_norms * (1 + surprise)
        
        # Determine sparsity ratio for this layer
        if dynamic_sparsity is not None:
            current_sparsity = dynamic_sparsity
        elif self.use_layer_specific and layer_name in self.layer_sparsity_ratios:
            current_sparsity = self.layer_sparsity_ratios[layer_name]
        else:
            current_sparsity = self.sparsity_ratio
        
        # Calculate number of elements to keep
        k = int(in_features * (1.0 - current_sparsity))
        k = max(1, min(k, in_features))  # Ensure k is valid
        
        # Get top-k indices based on WINA scores
        if len(wina_scores.shape) > 2:
            # Handle batch dimensions
            batch_shape = wina_scores.shape[:-1]
            flat_scores = wina_scores.view(-1, in_features)
            
            # Get top-k for each sample in batch
            _, top_k_indices = torch.topk(flat_scores, k, dim=-1)
            
            # Create mask
            mask = torch.zeros_like(flat_scores)
            mask.scatter_(-1, top_k_indices, 1.0)
            mask = mask.view(*batch_shape, in_features)
        else:
            # Simple case
            _, top_k_indices = torch.topk(wina_scores, k, dim=-1)
            mask = torch.zeros_like(wina_scores)
            mask.scatter_(-1, top_k_indices, 1.0)
        
        gated_states = hidden_states * mask
        if return_scores:
            return gated_states, wina_scores
        return gated_states
    
    def set_layer_sparsity(self, layer_name: str, sparsity_ratio: float):
        """Set layer-specific sparsity ratio."""
        self.layer_sparsity_ratios[layer_name] = sparsity_ratio
    
    def adaptive_sparsity_allocation(self, layer_names: List[str], global_sparsity: float):
        """
        Implement adaptive sparsity allocation using greedy algorithm.
        This prioritizes computational resources for more critical layers.
        
        Args:
            layer_names: List of layer names to configure
            global_sparsity: Target global sparsity level
        """
        # Simple heuristic: assign higher sparsity to later layers
        # In practice, this could be optimized based on layer importance
        num_layers = len(layer_names)
        for i, layer_name in enumerate(layer_names):
            # Gradually increase sparsity for later layers
            layer_sparsity = global_sparsity * (0.5 + 0.5 * i / max(1, num_layers - 1))
            self.set_layer_sparsity(layer_name, min(layer_sparsity, 0.8))  # Cap at 80%
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_ratio': self._cache_hits / max(1, self._cache_hits + self._cache_misses)
        }
    
    def clear_cache(self):
        """Clear the column norm cache."""
        self._column_norm_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

class MetaWINASparsifier(WINASparsifier):
    """
    Extends WINA sparsifier with meta-learning control that dynamically adjusts sparsity
    based on an auxiliary objective (e.g., prediction error or consistency).
    """
    def __init__(self, sparsity_ratio: float = 0.5, use_layer_specific: bool = True, control_dim: int = 64):
        super().__init__(sparsity_ratio, use_layer_specific)
        self.control_net = nn.Sequential(
            nn.Linear(control_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # Extended to 4 signals: sparsity interp, window, dilation, sparsity adjust
            nn.Sigmoid()
        )
        self.control_dim = control_dim
        
    def apply_wina_gating(self, hidden_states, weight_matrix, layer_name="default", cache_key=None, control_input=None):
        # If control input provided, compute dynamic sparsity
        dynamic_sparsity = None
        if control_input is not None:
            dynamic_sparsity = self.get_dynamic_sparsity(self.sparsity_ratio, control_input)
        
        # Call base method with dynamic sparsity
        base_gated = super().apply_wina_gating(hidden_states, weight_matrix, layer_name, cache_key, dynamic_sparsity)
        
        # If control input provided, use it to adjust sparsity
        if control_input is not None:
            control_signals = self.control_net(control_input)  # (batch_size, 4)
            control_signal = control_signals[:, 0]  # sparsity interpolation signal
            # Reshape control_signal to match hidden_states dimensions
            control_signal = control_signal.view(-1, *([1]*(hidden_states.dim()-1)))
            # Interpolate between original and gated based on control signal
            output = control_signal * base_gated + (1 - control_signal) * hidden_states
        else:
            output = base_gated
            
        return output

    def get_dynamic_sparsity(self, base_sparsity, control_input):
        control_signals = self.control_net(control_input)
        signal = control_signals[:, 3].mean().item()  # Use the 4th signal for sparsity adjustment
        return base_sparsity * (0.5 + signal)  # Adjust between 0.5*base and 1.5*base

    def get_dynamic_window(self, base_window, control_input):
        control_signals = self.control_net(control_input)
        signal = control_signals[:, 1].mean().item()
        return int(base_window * (0.5 + 1.5 * signal))

    def get_dynamic_dilation(self, base_dilation, control_input):
        control_signals = self.control_net(control_input)
        signal = control_signals[:, 2].mean().item()
        return int(base_dilation * (0.5 + 1.5 * signal))


class WINAAttention(nn.Module):
    """
    Multi-head attention with WINA sparse activation.
    
    This attention mechanism applies WINA sparsification at multiple stages:
    1. Input projections (Q, K, V)
    2. Attention weights
    3. Output projection
    
    This provides significant computational savings while maintaining
    better approximation quality than magnitude-only sparse methods.
    """
    
    def __init__(self,
                     d_model: int,
                     n_heads: int,
                     config: EnhancedCTMConfig,
                     dropout: float = 0.1,
                     sparsity_ratio: float = 0.5,
                     use_adaptive_sparsity: bool = True,
                     control_dim: int = 64):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.use_adaptive_sparsity = use_adaptive_sparsity

            self.config = config
            
            assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
            
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            
            self.dropout = nn.Dropout(dropout)
            self.wina_sparsifier = MetaWINASparsifier(sparsity_ratio=sparsity_ratio, control_dim=control_dim)
            # Setup adaptive sparsity if enabled
            if use_adaptive_sparsity:
                layer_names = ["query", "key", "value", "attention", "output"]
                self.wina_sparsifier.adaptive_sparsity_allocation(layer_names, sparsity_ratio)
            self.locality_predictor = nn.Linear(d_model, 2 * n_heads)
            self.control_proj = nn.Linear(d_model, control_dim)
    
            if self.config.ctm_selective_quantization:
                self.selective_quantizer = SelectiveQuantizer(
                    min_bits=self.config.ctm_quant_min_bits,
                    max_bits=self.config.ctm_quant_max_bits,
                )
            else:
                self.selective_quantizer = None


        
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
    
        control_input = self.control_proj(query.mean(dim=1))
         # Apply WINA sparsification to input representations
        query_sparse, query_scores = self.wina_sparsifier.apply_wina_gating(query, self.q_proj.weight, "query", f"q_proj_{id(self.q_proj.weight)}", control_input, return_scores=True)
        key_sparse, key_scores = self.wina_sparsifier.apply_wina_gating(key, self.k_proj.weight, "key", f"k_proj_{id(self.k_proj.weight)}", control_input, return_scores=True)
        value_sparse, value_scores = self.wina_sparsifier.apply_wina_gating(value, self.v_proj.weight, "value", f"v_proj_{id(self.v_proj.weight)}", control_input, return_scores=True)

        q_weight = self.q_proj.weight
        k_weight = self.k_proj.weight
        v_weight = self.v_proj.weight
        out_weight = self.out_proj.weight

        quantize = (self.config.quant_enabled_training and self.training) or \
                   (self.config.quant_enabled_inference and not self.training)

        if quantize and self.selective_quantizer:
            q_weight = self.selective_quantizer(q_weight, query_scores.mean(dim=[0, 1]))
            k_weight = self.selective_quantizer(k_weight, key_scores.mean(dim=[0, 1]))
            v_weight = self.selective_quantizer(v_weight, value_scores.mean(dim=[0, 1]))

        Q = F.linear(query_sparse, q_weight)
        K = F.linear(key_sparse, k_weight)
        V = F.linear(value_sparse, v_weight)
    
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
    
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
    
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
    
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    
        # Locality bias
        locality_params = self.locality_predictor(query.mean(dim=1)).view(batch_size, self.n_heads, 2)
    
        locality_sigma = locality_params[..., 0].mean()
    
        locality_strength = locality_params[..., 1].mean()
        # Add learnable locality bias to all heads
        positions = torch.arange(seq_len, device=query.device)
    
        dist = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
    
        locality_bias = torch.exp(-dist / locality_sigma.clamp(min=1e-5)) * locality_strength
    
        locality_bias = locality_bias.unsqueeze(0).unsqueeze(0)
    
        scores = scores + locality_bias
    
        if attention_mask is not None:
    
            scores = scores.masked_fill(~attention_mask.unsqueeze(1), float('-inf'))
    
        attention_weights = F.softmax(scores, dim=-1)
    
        identity_weight = torch.eye(attention_weights.shape[-1], device=attention_weights.device, dtype=attention_weights.dtype)
    
        attention_weights_sparse = self.wina_sparsifier.apply_wina_gating(
    
            attention_weights.view(-1, seq_len), identity_weight, "attention", "attention_wina", control_input
    
        ).view(attention_weights.shape)
    
        attention_weights_sparse = self.dropout(attention_weights_sparse)
    
        context = torch.matmul(attention_weights_sparse, V)
    
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
        context_sparse, context_scores = self.wina_sparsifier.apply_wina_gating(context, self.out_proj.weight, "output", f"out_proj_{id(self.out_proj.weight)}", control_input, return_scores=True)
        
        if quantize and self.selective_quantizer:
            out_weight = self.selective_quantizer(out_weight, context_scores.mean(dim=[0, 1]))

        output = F.linear(context_sparse, out_weight)
    
        return output
    
    
    def get_sparsity_stats(self) -> Dict[str, Any]:
        """Get sparsity statistics and cache performance."""
        return {
            'layer_sparsity_ratios': self.wina_sparsifier.layer_sparsity_ratios,
            'cache_stats': self.wina_sparsifier.get_cache_stats()
        }

class WINAEnhancedMLP(nn.Module):
    """
    MLP layer enhanced with WINA sparse activation.
    
    Applies WINA sparsification to both the intermediate and output layers
    for maximum computational efficiency while preserving model quality.
    """
    
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 sparsity_ratio: float = 0.5,
                 activation: str = "relu"):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.wina_sparsifier = WINASparsifier(sparsity_ratio=sparsity_ratio)
        
        # Setup layer-specific sparsity
        self.wina_sparsifier.set_layer_sparsity("intermediate", sparsity_ratio)
        self.wina_sparsifier.set_layer_sparsity("output", sparsity_ratio * 0.8)  # Less sparsity for output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply WINA to input before first linear layer
        x_sparse = self.wina_sparsifier.apply_wina_gating(
            x, self.linear1.weight, "intermediate", f"linear1_{id(self.linear1.weight)}"
        )
        # First linear transformation
        intermediate = self.linear1(x_sparse)
        intermediate = self.activation(intermediate)
        intermediate = self.dropout(intermediate)
        
        # Apply WINA to intermediate representation
        intermediate_sparse = self.wina_sparsifier.apply_wina_gating(
            intermediate, self.linear2.weight, "output", f"linear2_{id(self.linear2.weight)}"
        )
        
        # Second linear transformation
        output = self.linear2(intermediate_sparse)
        
        return output


class _EntropyProxyModel(nn.Module):
    """
    A learnable RNN-based model to estimate next-byte entropy and provide a
    training loss for itself.
    It predicts the probability distribution of the next byte and calculates entropy
    from this distribution.
    """
    def __init__(self,
                 byte_vocab_size: int = 256,
                 embedding_dim: int = 64,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.byte_vocab_size = byte_vocab_size
        self.embedding = nn.Embedding(byte_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, byte_vocab_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, byte_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            byte_sequence: Tensor of shape (batch_size, seq_len), dtype=torch.uint8.

        Returns:
            A tuple containing:
            - entropy_scores: Tensor of shape (batch_size, seq_len), dtype=torch.float32.
                              Entropy of the predicted next-byte distribution at each position.
            - aux_loss: Scalar tensor, the cross-entropy loss for next-byte prediction.
        """
        batch_size, seq_len = byte_sequence.shape
        device = byte_sequence.device

        # Embed byte sequence
        embedded = self.embedding(byte_sequence.long())  # (batch_size, seq_len, embedding_dim)

        # Get LSTM outputs for predicting the *next* byte
        # For predicting byte at t+1, we use LSTM output at time t
        # So, we pass the sequence up to seq_len-1 to predict bytes from 1 to seq_len
        lstm_input = embedded[:, :-1, :] # (batch_size, seq_len-1, embedding_dim)
        
        if lstm_input.shape[1] == 0: # Handle sequences of length 1
            # Cannot predict next byte for a sequence of length 1 in this setup
            # Return zero loss and zero/uniform entropy
            dummy_entropy = torch.zeros((batch_size, seq_len), device=device, dtype=torch.float32)
            # Uniform entropy for the single byte if needed, or just zero
            if seq_len > 0:
                 dummy_entropy[:,0] = -torch.log(torch.tensor(1.0/self.byte_vocab_size, device=device))

            return dummy_entropy, torch.tensor(0.0, device=device)

        lstm_out, _ = self.lstm(lstm_input)  # (batch_size, seq_len-1, hidden_dim)
        
        # Get logits for the next byte prediction
        # logits for byte_1, byte_2, ..., byte_{seq_len-1}
        next_byte_logits = self.fc(lstm_out) # (batch_size, seq_len-1, byte_vocab_size)

        # Calculate auxiliary loss for next-byte prediction
        # Targets are byte_sequence from 1 to seq_len
        targets = byte_sequence[:, 1:].long() # (batch_size, seq_len-1)
        
        # Reshape for CrossEntropyLoss: (N, C, d1, d2, ...) -> (batch_size * (seq_len-1), byte_vocab_size)
        # Targets: (N, d1, d2, ...) -> (batch_size * (seq_len-1))
        next_byte_logits_clamped = torch.clamp(next_byte_logits, min=-1e9, max=1e9)
        aux_loss = self.criterion(next_byte_logits_clamped.reshape(-1, self.byte_vocab_size), targets.reshape(-1))
        aux_loss = torch.nan_to_num(aux_loss, nan=0.0, posinf=0.0, neginf=0.0)

        # Calculate entropy scores from predicted probabilities
        # We need entropy for each position t based on prediction for x_t
        # The current next_byte_logits are for x_1, ..., x_{seq_len-1}
        # For x_0, we don't have a prediction from LSTM, assume uniform entropy or a fixed high value.
        
        probs = F.softmax(next_byte_logits, dim=-1) # (batch_size, seq_len-1, byte_vocab_size)
        entropy_from_predictions = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1) # (batch_size, seq_len-1)

        # Initialize entropy_scores for the full sequence length
        entropy_scores = torch.zeros((batch_size, seq_len), device=device, dtype=torch.float32)
        
        # Set entropy for the first byte (t=0) to a high value (e.g., log(vocab_size) for uniform)
        # This ensures the first byte can start a patch, similar to the original heuristic.
        entropy_scores[:, 0] = -torch.log(torch.tensor(1.0/self.byte_vocab_size, device=device))
        
        # Fill in entropies for t=1 to seq_len-1
        if seq_len > 1:
            entropy_scores[:, 1:] = entropy_from_predictions
            
        return entropy_scores, aux_loss

class DynamicEntropyPatcher(nn.Module): # Implements dynamic byte patching based on complexity (entropy).
    """
    Implements the dynamic, entropy-based patching and encoding mechanism
    inspired by the "Byte Latent Transformer" (BLT) paper.

    This module segments a raw byte sequence into variable-length patches based on
    next-byte entropy estimates, then encodes these patches into fixed-size vectors.
    """
    def __init__(self,
                 embedding_dim: int,
                 patch_cnn_channels: int,
                 patching_mode: str = "global",
                 global_threshold: float = 0.5,
                 relative_threshold: float = 0.1,
                 min_patch_size: int = 4,
                 max_patch_size: int = 128,
                 # New parameters for the learnable _EntropyProxyModel
                 entropy_byte_vocab_size: int = 256,
                 entropy_embedding_dim: int = 64,
                 entropy_hidden_dim: int = 128,
                 entropy_num_layers: int = 1,
                 entropy_dropout: float = 0.1):
        """
        Args:
            embedding_dim: The dimensionality of the output patch embeddings.
            patch_cnn_channels: The number of channels for the internal CNN encoder.
            patching_mode: The method for determining patch boundaries.
                           Options: "global", "relative_monotonic".
            global_threshold: The entropy threshold (θg) for the "global" mode.
            relative_threshold: The relative entropy increase threshold (θr) for
                                the "relative_monotonic" mode.
            min_patch_size: The minimum number of bytes in a patch.
            max_patch_size: The maximum number of bytes in a patch.
            entropy_byte_vocab_size: Vocab size for the entropy model's byte embeddings.
            entropy_embedding_dim: Embedding dimension for the entropy model.
            entropy_hidden_dim: Hidden dimension for the entropy model's RNN.
            entropy_num_layers: Number of RNN layers in the entropy model.
            entropy_dropout: Dropout rate for the entropy model's RNN.
        """
        super().__init__()
        if patching_mode not in ["global", "relative_monotonic"]:
            raise ValueError("patching_mode must be 'global' or 'relative_monotonic'")

        self.embedding_dim = embedding_dim
        self.patching_mode = patching_mode
        self.global_threshold = global_threshold
        self.relative_threshold = relative_threshold
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size

        self.entropy_model = _EntropyProxyModel(
            byte_vocab_size=entropy_byte_vocab_size,
            embedding_dim=entropy_embedding_dim,
            hidden_dim=entropy_hidden_dim,
            num_layers=entropy_num_layers,
            dropout=entropy_dropout
        )

        # A CNN-based encoder to map variable-length byte patches to fixed-size embeddings.
        self.patch_byte_encoder = nn.Sequential(
            # Input shape: (N, 1, max_patch_size)
            nn.Conv1d(in_channels=1, out_channels=patch_cnn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(patch_cnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(patch_cnn_channels, embedding_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            # Pool across the patch dimension to get a single vector per patch
            nn.AdaptiveAvgPool1d(1) # Output shape: (N, embedding_dim, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, byte_sequence: torch.Tensor) -> Tuple[torch.Tensor, List[List[Tuple[int, int]]], torch.Tensor]:
        """
        Processes a batch of byte sequences by patching and encoding them.

        Args:
            byte_sequence: A tensor of raw bytes.
                           Shape: (batch_size, seq_len_bytes), dtype=torch.uint8.

        Returns:
            A tuple containing:
            - encoded_patches: A tensor of patch embeddings, padded to the max number
                               of patches in the batch.
                               Shape: (batch_size, max_num_patches, embedding_dim).
            - patch_indices: A list of lists, where each inner list contains the
                             (start, end) byte indices for each patch in a sequence.
            - entropy_aux_loss: Scalar tensor, the auxiliary loss from the entropy model.
        """
        batch_size, seq_len = byte_sequence.shape
        device = byte_sequence.device

        # 1. Calculate entropy proxy scores and auxiliary loss for the entire batch
        entropy_scores, entropy_aux_loss = self.entropy_model(byte_sequence)  # (batch_size, seq_len), scalar

        all_patches_data = []
        all_patch_indices = []
        patches_per_sample = []  # To track how many patches each sample has

        # 2. Segment each sequence in the batch into patches
        for i in range(batch_size):
            single_entropy = entropy_scores[i]
            adaptive_threshold = single_entropy.mean() + 0.5 * single_entropy.std()
            patch_indices_for_sample = []
            current_start = 0

            while current_start < seq_len:
                # Search for a boundary within the allowed patch size window
                scan_start = current_start + self.min_patch_size
                scan_end = min(current_start + self.max_patch_size, seq_len)
                found_boundary_at = -1

                for t in range(scan_start, scan_end + 1):
                    if t >= seq_len: break

                    is_boundary = False
                    if self.patching_mode == "global":
                        if single_entropy[t] > adaptive_threshold:
                            is_boundary = True
                    elif self.patching_mode == "relative_monotonic":
                        # H(xt) - H(xt-1) > θr
                        if (single_entropy[t] - single_entropy[t-1]) > self.relative_threshold:
                            is_boundary = True
                    
                    if is_boundary:
                        found_boundary_at = t
                        break
                
                # Determine the final end of the patch
                if found_boundary_at != -1:
                    patch_end = found_boundary_at - 1 # End patch before the boundary
                else:
                    patch_end = min(current_start + self.max_patch_size - 1, seq_len - 1) # No boundary, take max size

                patch_end = max(patch_end, current_start) # Ensure patch has at least one byte
                
                all_patches_data.append(byte_sequence[i, current_start : patch_end + 1])
                patch_indices_for_sample.append((current_start, patch_end))
                current_start = patch_end + 1
            
            all_patch_indices.append(patch_indices_for_sample)
            patches_per_sample.append(len(patch_indices_for_sample))

        # Handle case where no patches were generated (e.g., empty input)
        if not all_patches_data:
            return torch.zeros((batch_size, 0, self.embedding_dim), device=device), all_patch_indices, entropy_aux_loss

        # 3. Prepare and encode all patches in a single batch for efficiency
        padded_patches = []
        for patch in all_patches_data:
            padded_patch = F.pad(patch, (0, self.max_patch_size - patch.shape[0]), 'constant', 0)
            padded_patches.append(padded_patch)
        
        patches_tensor = torch.stack(padded_patches)
        normalized_patches = patches_tensor.float() / 127.5 - 1.0 # Normalize to [-1, 1]
        patches_for_cnn = normalized_patches.unsqueeze(1) # Add channel dim
        
        encoded_patches_flat = self.patch_byte_encoder(patches_for_cnn).squeeze(-1)

        # 4. Re-assemble encoded patches into a padded batch tensor
        max_num_patches = max(patches_per_sample) if patches_per_sample else 0
        encoded_patches_list = torch.split(encoded_patches_flat, patches_per_sample)
        
        padded_batch_list = []
        for p_tensor in encoded_patches_list:
            num_pads = max_num_patches - p_tensor.shape[0]
            if num_pads > 0:
                padding = torch.zeros(num_pads, self.embedding_dim, device=device)
                p_tensor = torch.cat([p_tensor, padding], dim=0)
            padded_batch_list.append(p_tensor)

        if padded_batch_list:
            final_batch_tensor = torch.stack(padded_batch_list)
        else: # Handle empty input batch
            final_batch_tensor = torch.zeros((batch_size, 0, self.embedding_dim), device=device)
            
        return final_batch_tensor, all_patch_indices, entropy_aux_loss
from .enhanced_neuron_selection import EnhancedNeuronSelector #Enhances Nueron Selections with Biologically-Inspired Systems instead of Random
from .biological_neuron_selection import BiologicalNeuronSelector, BiologicalSelectionConfig
# Import original CTM modules to preserve exact behavior
# try:
from .modules import SynapseUNET, Squeeze, SuperLinear, LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding
from .utils import compute_normalized_entropy
from .constants import VALID_NEURON_SELECT_TYPES, VALID_POSITIONAL_EMBEDDING_TYPES
# except ImportError:
#     print("Warning: Could not import original CTM modules (e.g. from .modules). Using fallback implementations.")
#     SynapseUNET = None
#     SuperLinear = None
#     Squeeze = None
#     compute_normalized_entropy = None
#     VALID_NEURON_SELECT_TYPES = [ #Legacy
#     'first-last', 'random', 'random-pairing',
#     # Biologically-inspired types
#     'bio_hebbian', 'bio_plasticity', 'bio_competitive', 'bio_homeostatic',
#     'bio_evolutionary', 'bio_stdp', 'bio_criticality', 'bio_multi_objective',
#     # Hybrid approaches
#     'adaptive_random', 'performance_guided', 'task_aware']

# Consciousness Control System for Phase 0.5
class ConsciousnessController(nn.Module):
    """Controls the wake-up and sleep cycles of the CTM model."""

    def __init__(self, model_dim=512, max_attention_steps=100):
        super().__init__()
        self.model_dim = model_dim
        self.max_attention_steps = max_attention_steps
        self.attention_level = nn.Parameter(torch.tensor(0.0))  # Start at 0 (off)
        self.consciousness_state = 'sleeping'  # 'sleeping', 'waking', 'awake', 'sleeping_down'

        # Neural components for consciousness control
        self.attention_modulator = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, model_dim)
        )

        # Gentle wake-up lambda function
        self.wake_lambda = lambda step: min(1.0, (step / self.max_attention_steps) ** 0.5)
        # Gentle sleep lambda function
        self.sleep_lambda = lambda step: max(0.0, 1.0 - (step / self.max_attention_steps) ** 0.5)

    def wake_up(self, current_step):
        """Gradually wake up the model using lambda function."""
        if self.consciousness_state in ['sleeping', 'waking']:
            self.consciousness_state = 'waking'
            target_attention = self.wake_lambda(current_step)

            # Smooth transition to avoid stress
            with torch.no_grad():
                self.attention_level.fill_(target_attention) # Updated for PyTorch best practice

            if target_attention >= 1.0:
                self.consciousness_state = 'awake'
                print(f"  🌅 Model fully awakened (attention: {target_attention:.3f})")
            else:
                print(f"  🌄 Model waking up... (attention: {target_attention:.3f})")

            return target_attention
        return self.attention_level.item()

    def sleep_down(self, current_step):
        """Gradually put the model to sleep using lambda function."""
        if self.consciousness_state in ['awake', 'sleeping_down']:
            self.consciousness_state = 'sleeping_down'
            target_attention = self.sleep_lambda(current_step)

            # Smooth transition to avoid stress
            with torch.no_grad():
                self.attention_level.fill_(target_attention) # Updated for PyTorch best practice

            if target_attention <= 0.01:
                self.consciousness_state = 'sleeping'
                print(f"  🌙 Model fully asleep (attention: {target_attention:.3f})")
            else:
                print(f"  🌆 Model going to sleep... (attention: {target_attention:.3f})")

            return target_attention
        return self.attention_level.item()

    def get_attention_modulation(self):
        """Get the current attention modulation tensor."""
        attention_input = self.attention_level.unsqueeze(0)  # (1,)
        modulation = self.attention_modulator(attention_input)  # (model_dim,)
        return modulation

    def apply_consciousness_to_features(self, features):
        """Apply consciousness modulation to input features."""
        modulation = self.get_attention_modulation()
        # Broadcast modulation across batch and sequence dimensions
        if features.dim() == 3:  # (batch, seq, features)
            modulation = modulation.unsqueeze(0).unsqueeze(0)
        elif features.dim() == 2:  # (batch, features)
            modulation = modulation.unsqueeze(0)

        return features * modulation
    

class CTMFeedbackModule(nn.Module):
    """
    A feedback module designed to work with the CTM. It takes the CTM's
    final synchronization vector (thought vector) and generates a modulation
    signal for the diffusion process.
    This version adds a residual connection from the thought vector to the modulation signal to ensure more information is preserved.
    """
    def __init__(self, ctm_sync_dim: int, diffusion_model_dim: int, n_heads: int = 8):
        super().__init__()
        self.ctm_sync_dim = ctm_sync_dim
        self.diffusion_model_dim = diffusion_model_dim
        
        self.thought_projector = nn.Linear(ctm_sync_dim, diffusion_model_dim)
        self.residual_projector = nn.Linear(ctm_sync_dim, diffusion_model_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=diffusion_model_dim,
            num_heads=n_heads,
            batch_first=True
        )
        self.modulation_net = nn.Sequential(
            nn.Linear(diffusion_model_dim, diffusion_model_dim * 2),
            nn.GELU(),
            nn.Linear(diffusion_model_dim * 2, diffusion_model_dim)
        )
        self.norm = nn.LayerNorm(diffusion_model_dim)

    def forward(self, diffusion_state: torch.Tensor, ctm_thought_vector: torch.Tensor) -> torch.Tensor:
        """
        Produce a modulation signal based on the CTM's thought vector.
        """
        # Project the thought vector for attention
        query = self.thought_projector(ctm_thought_vector).unsqueeze(1)
        key_value = diffusion_state
        attn_output, _ = self.cross_attention(query, key_value, key_value)
        modulation = self.modulation_net(attn_output)
        
        # Project the thought vector for residual connection and add to modulation
        residual = self.residual_projector(ctm_thought_vector).unsqueeze(1)  # [B, 1, D]
        modulated_with_residual = modulation + residual
        
        return self.norm(modulated_with_residual)

import torch.nn as nn
import torch
import numpy as np
import math

from .modules import SynapseUNET, Squeeze, SuperLinear
from .utils import compute_normalized_entropy
from .constants import VALID_NEURON_SELECT_TYPES


class OriginalCTMCore(nn.Module):
    """
    Continuous Thought Machine (CTM).
    Adapted to use EnhancedCTMConfig.

    Technical report: https://arxiv.org/abs/2505.05522

    Interactive Website: https://pub.sakana.ai/ctm/

    Blog: https://sakana.ai/ctm/

    Thought takes time and reasoning is a process. 
    
    The CTM consists of three main ideas:
    1. The use of internal recurrence, enabling a dimension over which a concept analogous to thought can occur. 
    1. Neuron-level models, that compute post-activations by applying private (i.e., on a per-neuron basis) MLP 
       models to a history of incoming pre-activations.
    2. Synchronisation as representation, where the neural activity over time is tracked and used to compute how 
       pairs of neurons synchronise with one another over time. This measure of synchronisation is the representation 
       with which the CTM takes action and makes predictions.


    Args:
        iterations (int): Number of internal 'thought' ticks (T, in paper).
        d_model (int): Core dimensionality of the CTM's latent space (D, in paper).
                       NOTE: Note that this is NOT the representation used for action or prediction, but rather that which
                       is fully internal to the model and not directly connected to data.
        d_input (int): Dimensionality of input features `x` to the forward pass, also used as embed_dim for attention.
        heads (int): Number of attention heads.
        n_synch_out (int): Number of neurons used for output synchronisation (D_out, in paper).
        n_synch_action (int): Number of neurons used for action/attention synchronisation (D_action, in paper).
        synapse_depth (int): Depth of the synapse model (U-Net if > 1, else MLP).
        memory_length (int): History length for Neuron-Level Models (M, in paper).
        deep_nlms (bool): Use deeper (2-layer) NLMs if True, else linear.
                        NOTE: we almost always use deep NLMs, but a linear NLM is faster.
        memory_hidden_dims (int): Hidden dimension size for deep NLMs.
        do_layernorm_nlm (bool): Apply LayerNorm within NLMs.
                        NOTE: we never set this to true in the paper. If you set this to true you will get strange behaviour,
                        but you can potentially encourage more periodic behaviour in the dynamics. Untested; be careful.
        out_dims (int): Output dimension size.
                        NOTE: projected from synchronisation!
        prediction_reshaper (list): Shape for reshaping predictions before certainty calculation (task-specific).
                        NOTE: this is used to compute certainty and is needed when applying softmax for probabilities
        dropout (float): Dropout rate.
        dropout_nlm (float): Dropout rate for NLMs. If None, uses `dropout`.
        neuron_select_type (str): Neuron selection strategy ('first-last', 'random', 'random-pairing').
                        NOTE: some of this is legacy from our experimentation, but all three strategies are valid and useful. 
                            We dilineate exactly which strategies we use per experiment in the paper. 
                        - first-last: build a 'dense' sync matrix for output from the first D_out neurons and action from the 
                                      last D_action neurons. Flatten this matrix into the synchronisation representation. 
                                      This approach shares relationships for neurons and bottlenecks the gradients through them.
                                      NOTE: the synchronisation size will be (D_out/action * (D_out/action + 1))/2
                        - random: randomly select D_out neurons for the 'i' side pairings, and also D_out for the 'j' side pairings,
                                      also pairing those accross densely, resulting in a bottleneck roughly 2x as wide.
                                      NOTE: the synchronisation size will be (D_out/action * (D_out/action + 1))/2
                        - random-pairing (DEFAULT!): randomly select D_out neurons and pair these with another D_out neurons. 
                                      This results in much less bottlenecking and is the most up-to-date variant.
                                      NOTE: the synchronisation size will be D_out in this case; better control. 
        n_random_pairing_self (int): Number of neurons to select for self-to-self synch when random-pairing is used.
                        NOTE: when using random-pairing, i-to-i (self) synchronisation is rare, meaning that 'recovering a
                        snapshot representation' (see paper) is difficult. This alleviates that. 
                        NOTE: works fine when set to 0.                 


    """

    def __init__(self, config: EnhancedCTMConfig, neuromodulator_manager: Optional[NeuromodulatorManager] = None):
        super(OriginalCTMCore, self).__init__()
        self.config = config
        if config.enable_neuromodulators:
            self.neuromodulators = nn.ModuleDict({
                name: globals()[name.capitalize() + 'Modulator'](config.neuromodulator_dim)
                for name in config.active_neuromodulators
            })
            num_mods = len(self.neuromodulators)
            self.mod_fusion = nn.Linear(num_mods * config.neuromodulator_dim, config.ctm_d_model)

        # --- Core Parameters from Config ---
        self.iterations = config.ctm_iterations
        self.d_model = config.ctm_d_model  # CTM's internal latent space dimensionality
        self.d_input = config.ctm_input_dim # Dimensionality of external input features for attention
        self.memory_length = config.ctm_memory_length
        self.prediction_reshaper = config.ctm_prediction_reshaper
        self.n_synch_out = config.ctm_n_synch_out
        self.n_synch_action = config.ctm_n_synch_action
        self.out_dims = config.ctm_out_dims # CTM's direct output projection dim
        self.neuron_select_type = config.ctm_neuron_select_type
        
        # Resolve dropout_nlm from config
        dropout_nlm = config.ctm_dropout_nlm if config.ctm_dropout_nlm is not None else config.ctm_dropout
        
        # Other parameters from config needed for module setup
        heads = config.ctm_heads
        n_random_pairing_self = config.ctm_n_random_pairing_self
        synapse_depth = config.ctm_synapse_depth
        deep_nlms = config.ctm_deep_nlms
        do_layernorm_nlm = config.ctm_do_layernorm_nlm
        memory_hidden_dims = config.ctm_memory_hidden_dims
        dropout = config.ctm_dropout # General dropout for attention etc.

        # --- Assertions ---
        self.verify_args() # verify_args will use self.d_model, self.n_synch_out etc.
        
        # --- Predictive Coding Parameters ---
        self.use_predictive_coding = getattr(config, 'ctm_use_predictive_coding', True)
        self.pc_num_layers = getattr(config, 'ctm_pc_num_layers', 4)
        if self.use_predictive_coding:
            assert self.d_model % self.pc_num_layers == 0, "d_model must be divisible by pc_num_layers"
            pc_layer_size = self.d_model // self.pc_num_layers
            self.prediction_nets = nn.ModuleList(
                [nn.Linear(pc_layer_size, pc_layer_size) for _ in range(self.pc_num_layers - 1)]
            )

        # --- Activity-Dependent Plasticity Parameters ---
        self.use_activity_plasticity = getattr(config, 'use_activity_plasticity', True)
        self.plasticity_learning_rate = getattr(config, 'ctm_plasticity_learning_rate', 1e-4)
        self.plastic_synapses = nn.Linear(self.d_model, self.d_model, bias=False)
        nn.init.zeros_(self.plastic_synapses.weight) # Start with no plastic influence
        
        self.register_buffer('last_state_trace', torch.zeros((1, self.d_model, self.memory_length)), persistent=False)

        self.biological_selector = None
        if self.neuron_select_type.startswith('bio_'):
            try:
                # Create a config for the selector
                bio_config = BiologicalSelectionConfig(selection_type=self.neuron_select_type.replace('bio_', ''))
                self.biological_selector = BiologicalNeuronSelector(config=bio_config)
            except ImportError:
                print("Warning: biological_neuron_selection.py not found. Cannot use biological selector for plasticity.")
                self.biological_selector = None

        # --- Input Processing / Attention ---
        # q_proj projects synchronisation_action (related to CTM's d_model) to ctm_input_dim for attention query
        self.q_proj = nn.LazyLinear(self.d_input) if heads > 0 else None # This q_proj is for CTM's internal query generation
        
        # Instantiate the chosen attention mechanism
        if heads > 0:

            if config.attention_type == "binary_sparse":
                self.attention = BinarySparseAttention(
                    embed_dim=self.d_input,
                    num_heads=heads,
                    sparsity_ratio=config.sparse_attention_ratio,
                    binary_pattern_size=config.binary_pattern_size,
                    dropout=dropout
                )
            elif config.attention_type == "standard":
                self.attention = nn.MultiheadAttention(
                    embed_dim=self.d_input,
                    num_heads=heads,
                    dropout=dropout,
                    batch_first=True,
                    bias=config.attention_qkv_bias # Standard MHA also has a bias for qkv
                )
            elif config.attention_type == "WINA":
                self.attention = WINAAttention(
                    d_model=self.d_input,
                    n_heads=heads,
                    dropout=dropout,
                    sparsity_ratio=0.5,  # Adjustable
                    use_adaptive_sparsity=True
                )
            else:
                raise ValueError(f"Unsupported attention_type: {config.attention_type}")
        else:
            self.attention = None
        
        # --- Core CTM Modules ---
        # Synapses operate on CTM's internal d_model. Input to synapses is (attn_out + activated_state)
        # attn_out is self.d_input (from external features), activated_state is self.d_model.
        # The SynapseUNET or MLP within get_synapses should handle this combined input size.
        # Pass the correct combined input dimension and output dimension to synapses.
        synapse_input_dim = self.d_input + self.d_model  # Combined input size
        self.synapses = self.get_synapses(synapse_depth, synapse_input_dim, self.d_model, dropout)
        self.trace_processor = self.get_neuron_level_models(deep_nlms, do_layernorm_nlm, self.memory_length, memory_hidden_dims, self.d_model, dropout_nlm)

        #  --- Start States (depend on CTM's d_model) ---
        self.register_parameter('start_activated_state', nn.Parameter(torch.zeros((self.d_model)).uniform_(-math.sqrt(1/(self.d_model)), math.sqrt(1/(self.d_model)))))
        self.register_parameter('start_trace', nn.Parameter(torch.zeros((self.d_model, self.memory_length)).uniform_(-math.sqrt(1/(self.d_model+self.memory_length)), math.sqrt(1/(self.d_model+self.memory_length)))))

        # --- Synchronisation (depends on CTM's d_model and n_synch parameters) ---
        self.neuron_select_type_out, self.neuron_select_type_action = self.get_neuron_select_type()
        self.synch_representation_size_action = self.calculate_synch_representation_size(self.n_synch_action)
        self.synch_representation_size_out = self.calculate_synch_representation_size(self.n_synch_out)
        
        # print(f"Synch representation size action: {self.synch_representation_size_action}")
        # print(f"Synch representation size out: {self.synch_representation_size_out}")

        if self.synch_representation_size_action > 0:
            self.set_synchronisation_parameters('action', self.n_synch_action, n_random_pairing_self)
        # Always set for 'out', even if size is 0, to register buffers for indices if n_synch_out > 0
        self.set_synchronisation_parameters('out', self.n_synch_out, n_random_pairing_self)

        # --- Output Procesing (projects from CTM's synchronisation to ctm_out_dims) ---
        self.output_projector = nn.Sequential(nn.LazyLinear(self.out_dims))

        # --- Internal Feedback Module (Self-Modulation) ---
        self.use_internal_feedback = getattr(config, 'ctm_use_internal_feedback', True)
        if self.use_internal_feedback:
            self.internal_feedback_module = CTMFeedbackModule(
                ctm_sync_dim=self.d_model,
                diffusion_model_dim=self.d_model,
                n_heads=config.ctm_heads
            )
        else:
            self.internal_feedback_module = None

        # --- Bidirectional Reasoning Controller ---
        self.enable_bidirectional_reasoning = getattr(config, 'enable_bidirectional_reasoning', False)
        if self.enable_bidirectional_reasoning:
            self.reasoning_controller = torch.jit.script(BidirectionalReasoningController(
                d_model=self.d_model,
                sync_dim=self.synch_representation_size_out # Use output sync for reasoning control
            ))

        self.use_basal_ganglia = getattr(config, 'ctm_enable_basal_ganglia', True)
        if self.use_basal_ganglia and self.synch_representation_size_action > 0:
            self.basal_ganglia = BasalGangliaMechanism(
                d_model=self.d_model,
                action_dim=self.synch_representation_size_action,
                dopamine_dim=config.ctm_bg_dopamine_dim,
                context_dim=self.d_model
            )
        else:
            self.basal_ganglia = None
            self.compute_synchronisation = torch.compile(self.compute_synchronisation)
            self.forward_with_full_tracking = torch.compile(self.forward_with_full_tracking)


    # --- Core CTM Methods ---

    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        """
        Computes synchronisation to be used as a vector representation. 

        A neuron has what we call a 'trace', which is a history (time series) that changes with internal
        recurrence. i.e., it gets longer with every internal tick. There are pre-activation traces
        that are used in the NLMs and post-activation traces that, in theory, are used in this method. 

        We define sychronisation between neuron i and j as the dot product between their respective
        time series. Since there can be many internal ticks, this process can be quite compute heavy as it
        involves many dot products that repeat computation at each step. #Dot product replaced with Matmul computations. 
        
        Therefore, in practice, we update the synchronisation based on the current post-activations,
        which we call the 'activated state' here. This is possible because the inputs to synchronisation 
        are only updated recurrently at each step, meaning that there is a linear recurrence we can
        leverage. 

        Unified vectorized synchronization using masked matmul.
        """
        if synch_type == 'action':
            n_synch = self.n_synch_action
            neuron_indices_left = self.action_neuron_indices_left
            neuron_indices_right = self.action_neuron_indices_right
        elif synch_type == 'out':
            n_synch = self.n_synch_out
            neuron_indices_left = self.out_neuron_indices_left
            neuron_indices_right = self.out_neuron_indices_right
        elif synch_type == 'h':
            n_synch = self.n_synch_h
            neuron_indices_left = self.h_neuron_indices_left
            neuron_indices_right = self.h_neuron_indices_right

        B = activated_state.size(0)
        selected_left = activated_state[:, neuron_indices_left]  # (B, n_synch)
        selected_right = activated_state[:, neuron_indices_right]  # (B, n_synch)

        # Compute full outer product: (B, n_synch, n_synch)
        outer = torch.bmm(selected_left.unsqueeze(2), selected_right.unsqueeze(1))

        # Create mask based on type (computed once in init as buffer, but for simplicity here)
        mask = torch.zeros(B, n_synch, n_synch, device=activated_state.device)
        if self.neuron_select_type in ('first-last', 'random', 'bio_hebbian', 'bio_plasticity', 'bio_competitive', 'bio_homeostatic', 'bio_evolutionary', 'bio_stdp', 'bio_criticality', 'bio_multi_objective', 'adaptive_random', 'performance_guided', 'task_aware'):
            # Upper triangle including diagonal for full cross synchronization
            triu_mask = torch.triu(torch.ones(n_synch, n_synch, device=activated_state.device), diagonal=0)
            mask = triu_mask.unsqueeze(0).expand(B, -1, -1)
        else:  # 'random-pairing' etc. - assume pairs are aligned in indices
            # Diagonal for paired elements
            diag_mask = torch.eye(n_synch, device=activated_state.device)
            mask = diag_mask.unsqueeze(0).expand(B, -1, -1)

        # Apply mask and flatten to match original output shape
        masked_outer = outer * mask
        pairwise_product = masked_outer.view(B, -1)  # Flatten to (B, n_synch * n_synch), but only non-zero elements matter

        # Trim to actual size (original non-zero count)
        actual_size = self.calculate_synch_representation_size(n_synch)
        pairwise_product = pairwise_product[:, :actual_size]

        # Recurrent update (unchanged)
        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1

        synchronisation = decay_alpha / (torch.sqrt(decay_beta))
        return synchronisation, decay_alpha, decay_beta

    def compute_certainty(self, current_prediction):
        """
        Compute the certainty of the current prediction.
        
        We define certainty as being 1-normalised entropy.

        For legacy reasons we stack that in a 2D vector as this can be used for optimisation later.
        """
        B = current_prediction.size(0)
        reshaped_pred = current_prediction.reshape([B] + self.prediction_reshaper)
        ne = compute_normalized_entropy(reshaped_pred)
        current_certainty = torch.stack((ne, 1-ne), -1)
        return current_certainty

    # --- Setup Methods ---

    def get_neuron_level_models(self, deep_nlms, do_layernorm_nlm, memory_length, memory_hidden_dims, d_model, dropout):
        """
        Neuron level models are one of the core innovations of the CTM. They apply separate MLPs/linears to 
        each neuron.
        NOTE: the name 'SuperLinear' is largely legacy, but its purpose is to apply separate linear layers
            per neuron. It is sort of a 'grouped linear' function, where the group size is equal to 1. 
            One could make the group size bigger and use fewer parameters, but that is future work.

        NOTE: We used GLU() nonlinearities because they worked well in practice. 
        """
        if deep_nlms:
            return nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2 * memory_hidden_dims, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    SuperLinear(in_dims=memory_hidden_dims, out_dims=2, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )
        else:
            return nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )

    def get_synapses(self, synapse_depth, input_dim, output_dim, dropout):
        """
        The synapse model is the recurrent model in the CTM. It's purpose is to share information
        across neurons. If using depth of 1, this is just a simple single layer with nonlinearity and layernorm.
        For deeper synapse models we use a U-NET structure with many skip connections. In practice this performs
        better as it enables multi-level information mixing.

        The intuition with having a deep UNET model for synapses is that the action of synaptic connections is
        not necessarily a linear one, and that approximate a synapse 'update' step in the brain is non trivial.
        Hence, we set it up so that the CTM can learn some complex internal rule instead of trying to approximate
        it ourselves.
        
        Args:
            synapse_depth: Depth of the synapse model
            input_dim: Input dimension (d_input + d_model for combined attn_out + activated_state)
            output_dim: Output dimension (d_model for CTM's internal state)
            dropout: Dropout rate
        """
        if synapse_depth == 1:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.LazyLinear(output_dim * 2),  # Use LazyLinear to automatically handle input_dim
                nn.GLU(),  # GLU reduces dimension by half, so output_dim * 2 -> output_dim
                nn.LayerNorm(output_dim)
            )
        else:
            # SynapseUNET uses LazyLinear internally and handles variable input size correctly
            # It expects output_dim as the target output dimension and will adapt to input_dim automatically
            return SynapseUNET(output_dim, synapse_depth, 16, dropout)

    def set_synchronisation_parameters(self, synch_type: str, n_synch: int, n_random_pairing_self: int = 0):
        """
        1. Set the buffers for selecting neurons so that these indices are saved into the model state_dict.
        2. Set the parameters for learnable exponential decay when computing synchronisation between all
            neurons.
        """
        assert synch_type in ('out', 'action', 'h'), f"Invalid synch_type: {synch_type}"
        left, right = self.initialize_left_right_neurons(synch_type, self.d_model, n_synch, n_random_pairing_self)
        if synch_type == 'action':
            synch_representation_size = self.synch_representation_size_action
        elif synch_type == 'out':
            synch_representation_size = self.synch_representation_size_out
        else: # 'h'
            synch_representation_size = self.synch_representation_size_h
        self.register_buffer(f'{synch_type}_neuron_indices_left', left)
        self.register_buffer(f'{synch_type}_neuron_indices_right', right)
        self.register_parameter(f'decay_params_{synch_type}', nn.Parameter(torch.zeros(synch_representation_size), requires_grad=True))

    def initialize_left_right_neurons(self, synch_type, d_model, n_synch, n_random_pairing_self=0):
        """
        Initialize the left and right neuron indices based on the neuron selection type.
        This complexity is owing to legacy experiments, but we retain that these types of
        neuron selections are interesting to experiment with.
        Uses EnhancedNeuronSelector for biological and hybrid types.
        """
        device = self.start_activated_state.device

        # Ensure _enhanced_selector is initialized (it should be by get_neuron_select_type called in __init__)
        if not hasattr(self, '_enhanced_selector'):
             # Fallback initialization if somehow not set, though get_neuron_select_type should handle this.
             self._enhanced_selector = EnhancedNeuronSelector(neuron_select_type=self.neuron_select_type)

        if self.neuron_select_type.startswith('bio_') or \
           self.neuron_select_type in ['adaptive_random', 'performance_guided', 'task_aware']:
            # Use EnhancedNeuronSelector for these types.
            # Pass activations=None as they are not available during initialization.
            # The selector should handle this (e.g., by falling back to random or a default strategy).
            neuron_indices_left, neuron_indices_right, _ = self._enhanced_selector.select_neurons_for_synchronization( # The sync_mode is not currently used here, so we can ignore it with _ #The original dot product processing here was replaced with a torch matmul method.
                activations=None,
                synch_type=synch_type,
                n_synch=n_synch,
                d_model=d_model,
                targets=None,
                weights=None
            )
        elif self.neuron_select_type=='first-last':
            if synch_type == 'out':
                neuron_indices_left = neuron_indices_right = torch.arange(0, n_synch, device=device)
            elif synch_type == 'action':
                neuron_indices_left = neuron_indices_right = torch.arange(d_model-n_synch, d_model, device=device)

        elif self.neuron_select_type=='random':
            # Ensure replace=False for unique neuron selection
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch, replace=False)).to(device)
            neuron_indices_right = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch, replace=False)).to(device)

        elif self.neuron_select_type=='random-pairing':
            assert n_synch > n_random_pairing_self, f"Need at least {n_random_pairing_self} pairs for {self.neuron_select_type}"
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch, replace=False)).to(device)
            # Ensure replace=False for the second part as well for unique neurons
            neuron_indices_right_random_part = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch-n_random_pairing_self, replace=False)).to(device)
            neuron_indices_right = torch.concatenate((neuron_indices_left[:n_random_pairing_self], neuron_indices_right_random_part))
        else:
            raise ValueError(f"Unhandled neuron selection type in initialize_left_right_neurons: {self.neuron_select_type}")

        return neuron_indices_left.to(device), neuron_indices_right.to(device)

    def get_neuron_select_type(self):
        """
        Enhanced version that supports biological selection methods.
        """
        # Create enhanced selector if not exists
        if not hasattr(self, '_enhanced_selector'):
            # Ensure self.neuron_select_type is available from config
            # It's set in __init__ from config.ctm_neuron_select_type
            self._enhanced_selector = EnhancedNeuronSelector(
                neuron_select_type=self.neuron_select_type
            )
        
        return self._enhanced_selector.get_neuron_select_type()

    # --- Utilty Methods ---

    def verify_args(self):
        """
        Verify the validity of the input arguments to ensure consistent behaviour. 
        Specifically when selecting neurons for sychronisation using 'first-last' or 'random',
        one needs the right number of neurons
        """
        assert self.neuron_select_type in VALID_NEURON_SELECT_TYPES, \
            f"Invalid neuron selection type: {self.neuron_select_type}"
        
        if self.neuron_select_type == 'first-last':
            assert self.d_model >= (self.n_synch_out + self.n_synch_action), \
                "d_model must be >= n_synch_out + n_synch_action for neuron subsets"

    def calculate_synch_representation_size(self, n_synch):
        """
        Calculate the size of the synchronisation representation based on neuron selection type.
        """
        if self.neuron_select_type == 'random-pairing' or \
           self.neuron_select_type.startswith('bio_') or \
           self.neuron_select_type in ['adaptive_random', 'performance_guided', 'task_aware']:
            # For these types, n_synch neurons are selected and paired, resulting in n_synch synchronization values.
            synch_representation_size = n_synch
        elif self.neuron_select_type in ('first-last', 'random'):
            # For these, a dense matrix of n_synch x n_synch is formed, and upper triangle is taken.
            synch_representation_size = (n_synch * (n_synch + 1)) // 2
        else:
            raise ValueError(f"Unhandled neuron selection type in calculate_synch_representation_size: {self.neuron_select_type}")
        return synch_representation_size

    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device
        quantize = (self.config.quant_enabled_training and self.training) or \
                   (self.config.quant_enabled_inference and not self.training)

        if quantize and self.config.ctm_use_qat:
            x = self.quant(x)

        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []
        dopamine_error_tracking = []

        # --- Input Features (x is assumed to be pre-processed) ---
        # x has shape (B, S, d_input) where S is sequence length of features
        kv = x 

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1) # Shape: (B, d_model, memory_length)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1) # Shape: (B, d_model)

        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=torch.float32)

        # --- Initialise Recurrent Synch Values  ---
        decay_alpha_action, decay_beta_action = None, None
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        r_action, r_out = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1), torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)

        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')
        
        # --- Recurrent Loop  ---
        for stepi in range(self.iterations):

            # --- Calculate Synchronisation for Input Data Interaction ---
            synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action')

            # --- Basal Ganglia Gating ---
            if self.basal_ganglia is not None:
                action_gate, dopamine_error = self.basal_ganglia(
                    thought_vector=activated_state,
                    context=activated_state,
                    reward_signal=None
                )
                synchronisation_action = synchronisation_action * action_gate
                if track:
                    dopamine_error_tracking.append(dopamine_error.detach().cpu().numpy())

            # --- Interact with Data via Attention ---
            if self.attention is not None and self.q_proj is not None:
                q = self.q_proj(synchronisation_action).unsqueeze(1) # q shape: (B, 1, d_input)
                attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True) # attn_out shape: (B, 1, d_input)
                attn_out = attn_out.squeeze(1) # attn_out shape: (B, d_input)
                pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1) # (B, d_input + d_model)
            else: # No attention mechanism
                attn_weights = None # For tracking
                # If no attention, synapse input might just be activated_state or require different handling
                # For now, let's assume if attention is None, we might pass activated_state directly or a zero tensor for attn_out part
                # This part needs clarification if heads=0 is a valid use case for this modified CTM.
                # Assuming for now that if attention is None, this path might not be fully defined by original logic.
                # Simplest assumption: if no attention, perhaps no external input interaction this way.
                # Let's make pre_synapse_input just activated_state if no attention.
                # This would change the input dim to synapses. LazyLinear would adapt.
                # However, the original code structure implies attention is central.
                # A more robust way if heads=0:
                # pre_synapse_input = activated_state # This would be (B, d_model)
                # Or, if concatenation is always expected:
                zero_attn_out_replacement = torch.zeros(B, self.d_input, device=device, dtype=activated_state.dtype)
                pre_synapse_input = torch.concatenate((zero_attn_out_replacement, activated_state), dim=-1)


            # --- Apply Synapses ---
            state = self.synapses(pre_synapse_input) # state shape: (B, d_model)
            
            # --- Apply Activity-Dependent Plasticity ---
            if self.use_activity_plasticity:
                plastic_adjustment = self.plastic_synapses(activated_state)
                state = state + plastic_adjustment

            # The 'state_trace' is the history of incoming pre-activations
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1) # (B, d_model, memory_length)

            # --- Apply Neuron-Level Models ---
            activated_state = self.trace_processor(state_trace) # (B, d_model)

            # Apply Neuromodulators
            if self.config.enable_neuromodulators and hasattr(self, 'neuromodulators'):
                mod_outputs = [mod(activated_state) for mod in self.neuromodulators.values()]
                if mod_outputs:
                    concatenated_mods = torch.cat(mod_outputs, dim=-1)
                    fused_mod = self.mod_fusion(concatenated_mods)
                    activated_state = activated_state * fused_mod

            # --- Apply Internal CTM Feedback (Self-Modulation) ---
            if self.use_internal_feedback and self.internal_feedback_module is not None:
                # The current state provides the query (thought) and key/value (context)
                feedback_signal = self.internal_feedback_module(
                    diffusion_state=activated_state.unsqueeze(1), # (B, 1, D)
                    ctm_thought_vector=activated_state # (B, D)
                )
                # Add the feedback to the current state
                activated_state = activated_state + feedback_signal.squeeze(1)

            # --- Calculate Synchronisation for Output Predictions ---
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # --- Tracking ---
            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                if attn_weights is not None:
                    attention_tracking.append(attn_weights.detach().cpu().numpy())
                else: # Handle case where attn_weights might not be generated
                    attention_tracking.append(None) 
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())

        # --- Return Values ---
        if quantize and self.config.ctm_use_qat:
            predictions = self.dequant(predictions)
        if track:
            tracking_results = (np.array(synch_out_tracking), np.array(synch_action_tracking)), \
                               np.array(pre_activations_tracking), np.array(post_activations_tracking), \
                               np.array(attention_tracking)
            if dopamine_error_tracking:
                return predictions, certainties, tracking_results, np.array(dopamine_error_tracking)
            return predictions, certainties, tracking_results
        return predictions, certainties, synchronisation_out
    def compute_predictive_coding_loss(self, activated_state: torch.Tensor) -> torch.Tensor:
        """
        Computes the predictive coding loss based on hierarchical layers
        within the CTM's activated state.
        Higher layers predict the state of lower layers.
        """
        if not self.use_predictive_coding:
            return torch.tensor(0.0, device=activated_state.device)

        # Split the activated state into hierarchical layers
        # layers are ordered from low to high, e.g., layers[0] is the lowest
        try:
            layers = torch.chunk(activated_state, self.pc_num_layers, dim=1)
        except RuntimeError as e:
            # This can happen if activated_state cannot be split into pc_num_layers
            print(f"Warning: Could not compute predictive coding loss. Activated state shape {activated_state.shape} could not be chunked into {self.pc_num_layers} layers. Error: {e}")
            return torch.tensor(0.0, device=activated_state.device)
        
        total_pc_loss = 0.0
        
        # Higher layers predict lower layers
        for i in range(self.pc_num_layers - 1):
            higher_layer_idx = i + 1
            lower_layer_idx = i
            
            # Prediction from higher to lower layer
            # self.prediction_nets[i] predicts layer i from layer i+1
            predicted_lower_layer = self.prediction_nets[i](layers[higher_layer_idx])
            
            # Calculate prediction error (local loss)
            # We detach the target to prevent gradients from flowing back from the target,
            # ensuring the higher layer is trained to predict the lower, not the other way around.
            local_loss = F.mse_loss(predicted_lower_layer, layers[lower_layer_idx].detach())
            total_pc_loss += local_loss
            
        return total_pc_loss


    def forward_with_full_tracking(self, kv_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass that returns ALL CTM internal states for diffusion control.
        
        Returns:
            Dict containing:
            - predictions: (B, out_dims, iterations)
            - certainties: (B, 2, iterations)
            - sync_out_history: List of sync outputs per iteration
            - sync_action_history: List of sync actions per iteration
            - activated_states: List of activated states per iteration
            - state_traces: List of state traces per iteration
            - predictive_coding_loss: Accumulated predictive coding loss
        """
        B = kv_features.size(0)
        device = kv_features.device
        
        # Initialize tracking dictionaries and lists
        tracking_data = {
            'sync_out_history': [], 'sync_action_history': [], 'activated_states': [], 'state_traces': [],
            'attention_weights': [], 'pc_losses': [], 'dopamine_errors': [], 'plastic_adjustments': []
        }
        full_state_history = []

        # Initialize recurrent state
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)
        
        # Initialize synch values
        decay_alpha_action, decay_beta_action = None, None
        decay_alpha_out, decay_beta_out = None, None
        r_action = (torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1) if hasattr(self, 'decay_params_action') else None)
        r_out = torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)

        # Initial loop parameters
        step_pointer = 0
        total_steps_taken = 0
        max_steps = self.config.max_reasoning_steps if self.enable_bidirectional_reasoning else self.iterations

        # --- DYNAMIC BIDIRECTIONAL REASONING LOOP ---
        certainty_history = torch.tensor([], device=device)
        while total_steps_taken < max_steps:
            # Store the current state before processing the step
            full_state_history.append({
                'state_trace': state_trace, 'activated_state': activated_state, 'decay_alpha_action': decay_alpha_action,
                'decay_beta_action': decay_beta_action, 'decay_alpha_out': decay_alpha_out, 'decay_beta_out': decay_beta_out
            })

            # --- Main CTM Step Logic ---
            if r_action is not None:
                synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(activated_state, decay_alpha_action, decay_beta_action, r_action, 'action')
                if self.basal_ganglia:
                    action_gate, dopamine_error = self.basal_ganglia(activated_state, activated_state, None)
                    synchronisation_action = synchronisation_action * action_gate
                    tracking_data['dopamine_errors'].append(dopamine_error)
                tracking_data['sync_action_history'].append(synchronisation_action.clone())
                if self.attention:
                    q = self.q_proj(synchronisation_action).unsqueeze(1)
                    attn_out, attn_weights = self.attention(q, kv_features, kv_features, average_attn_weights=False, need_weights=True)
                    pre_synapse_input = torch.cat((attn_out.squeeze(1), activated_state), dim=-1)
                    tracking_data['attention_weights'].append(attn_weights.clone())
                else:
                    pre_synapse_input = torch.cat((kv_features.mean(dim=1), activated_state), dim=-1)
            else:
                 pre_synapse_input = torch.cat((kv_features.mean(dim=1), activated_state), dim=-1)

            state = self.synapses(pre_synapse_input)
            if self.use_activity_plasticity:
                plastic_adjustment = self.plastic_synapses(activated_state)
                state = state + plastic_adjustment
                tracking_data['plastic_adjustments'].append(plastic_adjustment.clone())

            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
            activated_state = self.trace_processor(state_trace)
            if self.use_internal_feedback:
                feedback = self.internal_feedback_module(activated_state.unsqueeze(1), activated_state)
                activated_state = activated_state + feedback.squeeze(1)
            
            if self.use_predictive_coding:
                tracking_data['pc_losses'].append(self.compute_predictive_coding_loss(activated_state))
            
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, 'out')
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)
            mean_conf = current_certainty[:,1].mean()
            certainty_history = torch.cat([certainty_history, mean_conf.unsqueeze(0)])
            if len(certainty_history) >= 3 and abs(certainty_history[-1] - certainty_history[-2]) < 0.01 and abs(certainty_history[-2] - certainty_history[-3]) < 0.01:
                break
            
            # --- Reasoning Control ---
            if self.enable_bidirectional_reasoning:
                step_delta, term_prob = self.reasoning_controller(activated_state, synchronisation_out)
                
                # Check for termination
                if (term_prob > self.config.reasoning_step_gating_threshold).all():
                    break
                
                # Update step pointer
                step_pointer = step_pointer + int(step_delta.mean().item())
                step_pointer = max(0, min(step_pointer, len(full_state_history) - 1)) # Clamp pointer

                # If moving backward, restore state from history
                if step_delta.mean().item() < 0:
                    restored_state = full_state_history[step_pointer]
                    state_trace, activated_state = restored_state['state_trace'], restored_state['activated_state']
                    decay_alpha_action, decay_beta_action = restored_state['decay_alpha_action'], restored_state['decay_beta_action']
                    decay_alpha_out, decay_beta_out = restored_state['decay_alpha_out'], restored_state['decay_beta_out']
            else:
                 step_pointer += 1 # Default linear progression

            total_steps_taken += 1
            if not self.enable_bidirectional_reasoning and total_steps_taken >= self.iterations:
                break

        # Collect final results from the last valid state
        final_state_data = full_state_history[-1]
        final_sync_out, _, _ = self.compute_synchronisation(final_state_data['activated_state'], final_state_data['decay_alpha_out'], final_state_data['decay_beta_out'], r_out, 'out')
        predictions = self.output_projector(final_sync_out)
        certainties = self.compute_certainty(predictions)

        # Confidence Thresholding
        abstain_mask = torch.zeros(B, dtype=torch.bool, device=device)
        if self.config.confidence_threshold > 0:
            confidence_scores = certainties[:, 1]  # Shape: (B,)
            abstain_mask = confidence_scores < self.config.confidence_threshold
        
        self.last_state_trace = final_state_data['state_trace'].detach()

        # Reshape predictions and certainties to be compatible with downstream logic
        final_predictions = predictions.unsqueeze(-1)
        final_certainties = certainties.unsqueeze(-1)
        if quantize and self.config.ctm_use_qat:
            predictions = self.dequant(predictions)

        return {
            'predictions': predictions,
            'certainties': certainties,
            'abstained': abstain_mask.unsqueeze(-1),
            'final_sync_out': synchronisation_out,
            'predictive_coding_loss': torch.stack(tracking_data['pc_losses']).mean() if tracking_data['pc_losses'] else torch.tensor(0.0, device=device),
            'dopamine_loss': torch.stack(tracking_data['dopamine_errors']).mean() if tracking_data['dopamine_errors'] else torch.tensor(0.0, device=device),
            **tracking_data
        }

class BidirectionalReasoningController(nn.Module):
    """
    A JIT-compiled controller that decides the direction of the CTM's reasoning process.
    It can decide to move forward, backward, or terminate the thought process
    based on the current state's confidence/coherence.
    """
    def __init__(self, d_model: int, sync_dim: int):
        super().__init__()
        self.d_model = d_model
        self.sync_dim = sync_dim
        # Input features are the activated state and the synchronization output
        controller_input_dim = d_model + sync_dim

        self.reasoning_gate = nn.Sequential(
            nn.Linear(controller_input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3), # Logits for [BACKWARD, STAY, FORWARD]
        )
        
        self.termination_gate = nn.Sequential(
            nn.Linear(controller_input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, activated_state: torch.Tensor, sync_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            activated_state: The CTM's current activated state (B, d_model)
            sync_out: The CTM's current synchronization output (B, sync_dim)
        
        Returns:
            step_delta: A tensor with values in {-1, 0, 1} indicating the step direction.
            termination_prob: A scalar tensor (0-1) indicating the probability of terminating.
        """
        controller_input = torch.cat([activated_state, sync_out], dim=-1)

        direction_logits = self.reasoning_gate(controller_input)
        # Gumbel-Softmax for sampling a discrete action (backward, stay, forward)
        direction_samples = F.gumbel_softmax(direction_logits, tau=1.0, hard=True)  # (B, 3)
        # Convert one-hot samples to step delta: [-1, 0, 1]
        step_values = torch.tensor([-1.0, 0.0, 1.0], device=activated_state.device)
        step_delta = torch.sum(direction_samples * step_values.view(1, 3), dim=1) # (B,)
         # Decide termination
        termination_prob = self.termination_gate(controller_input).squeeze(-1)# (B,)
        
        return step_delta, termination_prob


class GoalPredictor(nn.Module):
    """Predicts likely internal goals of observed agents"""
    def __init__(self, d_model: int, goal_dim: int = 8):
        super().__init__()
        self.d_model = d_model
        self.goal_dim = goal_dim
        self.goal_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, goal_dim)
        )
        self.goal_update = nn.GRU(d_model, goal_dim, batch_first=True)
        # Integrate neuromodulators for biological-like goal prediction
        self.neuromodulators = nn.ModuleDict({
            'dopamine': DopamineModulator(d_model),
            'serotonin': SerotoninModulator(d_model),
            'norepinephrine': NorepinephrineModulator(d_model)
        })
        
    def forward(self, current_state: torch.Tensor, prev_goal: torch.Tensor, reward_error: Optional[torch.Tensor] = None, uncertainty: Optional[torch.Tensor] = None, novelty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            current_state: Current neural state [batch, seq_len, d_model]
            prev_goal: Previous goal state [batch, seq_len, goal_dim]
            reward_error: Optional reward prediction error for dopamine
            uncertainty: Optional uncertainty for serotonin
            novelty: Optional novelty signal for norepinephrine
            
        Returns:
            Predicted goal state [batch, seq_len, goal_dim]
        """
        # Predict goal from current state and context
        context = current_state.mean(dim=1, keepdim=True).expand(-1, current_state.size(1), -1)
        goal_input = torch.cat([current_state, context], dim=-1)
        goal_pred = self.goal_net(goal_input)
        
        # Apply neuromodulators
        modulation = torch.ones_like(goal_pred)
        if reward_error is not None:
            modulation *= self.neuromodulators['dopamine'](goal_pred, reward_error)
        if uncertainty is not None:
            modulation *= self.neuromodulators['serotonin'](goal_pred, uncertainty)
        if novelty is not None:
            modulation *= self.neuromodulators['norepinephrine'](goal_pred, novelty)
        goal_pred = goal_pred * modulation
        
        # Update goal state using GRU
        updated_goal, _ = self.goal_update(goal_pred, prev_goal.unsqueeze(0))
        return updated_goal.squeeze(0)

class EmotionStateTracker(nn.Module):
    """Tracks and updates emotion states based on neural activity"""
    def __init__(self, d_model: int, num_emotion_dim: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_emotion_dim = num_emotion_dim
        self.emotion_proj = nn.Linear(d_model, num_emotion_dim)
        self.emotion_update = nn.GRU(num_emotion_dim, num_emotion_dim, batch_first=True)
        self.amygdala = AmygdalaSimulator(num_emotion_dim)
        # Integrate neuromodulators for biological-like emotion processing
        self.neuromodulators = nn.ModuleDict({
            'oxytocin': OxytocinModulator(num_emotion_dim),
            'endorphins': EndorphinsModulator(num_emotion_dim),
            'cortisol': CortisolModulator(num_emotion_dim)
        })
        
    def forward(self, neural_state: torch.Tensor, prev_emotion: torch.Tensor, social_context: Optional[torch.Tensor] = None, penalty: Optional[torch.Tensor] = None, threat_level: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            neural_state: Current neural state [batch, seq_len, d_model]
            prev_emotion: Previous emotion state [batch, seq_len, num_emotion_dim]
            social_context: Optional social context for oxytocin
            penalty: Optional penalty signal for endorphins
            threat_level: Optional threat level for cortisol
            
        Returns:
            Updated emotion state [batch, seq_len, num_emotion_dim]
        """
        # Project neural state to emotion space
        current_emotion = self.emotion_proj(neural_state)
        
        # Apply neuromodulators
        modulation = torch.ones_like(current_emotion)
        if social_context is not None:
            modulation *= self.neuromodulators['oxytocin'](current_emotion, social_context)
        if penalty is not None:
            modulation *= self.neuromodulators['endorphins'](current_emotion, penalty)
        if threat_level is not None:
            modulation *= self.neuromodulators['cortisol'](current_emotion, threat_level)
        current_emotion = current_emotion * modulation
        
        # Update emotion state using GRU
        updated_emotion, _ = self.emotion_update(current_emotion, prev_emotion.unsqueeze(0))
        return self.amygdala(updated_emotion.squeeze(0))
    
class AmygdalaSimulator(nn.Module):
    def __init__(self, emotion_dim):
        super().__init__()
        self.intensity_proj = nn.Linear(emotion_dim, 1)
        self.response_net = nn.Linear(emotion_dim, emotion_dim)
        self.threshold = nn.Parameter(torch.tensor(0.8))
        # Integrate neuromodulators for biological emotion regulation
        self.serotonin = SerotoninModulator(emotion_dim)
        self.cortisol = CortisolModulator(emotion_dim)
        self.gaba = GABAModulator(emotion_dim)

    def forward(self, emotion_state):
        intensity = torch.sigmoid(self.intensity_proj(torch.abs(emotion_state).mean(dim=-1, keepdim=True)))
        response = torch.tanh(self.response_net(emotion_state)) * (intensity > self.threshold).float()
        
        # Apply neuromodulators based on intensity
        uncertainty = intensity  # Use intensity as proxy for uncertainty/stress
        emotion_state = self.serotonin(emotion_state, uncertainty)
        emotion_state = self.cortisol(emotion_state, uncertainty)
        emotion_state = self.gaba(emotion_state, torch.norm(emotion_state, dim=-1))
        
        emotion_state = emotion_state + response * 0.1
        return emotion_state
class SynapticEmpathy(nn.Module):
    """
    Simulates a low-level, neuron-based form of empathy.

    This module operates directly on the neural activation histories (state traces)
    of the agent and an observed agent. High similarity in these traces is
    interpreted as "neural resonance," which in turn generates a reward signal and
    a modulation vector to influence the agent's own neural dynamics. This
    mechanism models a primitive, subconscious form of empathy based on shared
    neural patterns, rather than high-level cognitive interpretation.

    Args:
        d_model (int): The core dimensionality of the CTM's latent space.
        memory_length (int): The history length (M) of the neural activation traces.
        n_heads (int): The number of attention heads for the internal mirroring mechanism.
        dropout (float): Dropout rate for the attention mechanism.
    """
    def __init__(self, d_model: int, memory_length: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.memory_length = memory_length
        self.n_heads = n_heads

        # Computes neural resonance based on the similarity of activation traces
        # Takes absolute difference of traces and outputs a per-neuron score
        self.resonance_computer = nn.Sequential(
            nn.Linear(memory_length, memory_length // 2),
            nn.ReLU(),
            nn.Linear(memory_length // 2, 1)
        )

        # A cross-attention mechanism to map observed neural patterns to self-patterns
        self.mirroring_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Generates a synaptic modulation matrix based on empathic resonance
        self.synaptic_modulator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Tanh() # Output a modulation factor between -1 and 1
        )
        
        # A simple reward function based on positive resonance
        self.reward_generator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        # Integrate neuromodulators for enhanced empathy processing
        self.neuromodulators = nn.ModuleDict({
            'oxytocin': OxytocinModulator(d_model),
            'serotonin': SerotoninModulator(d_model)
        })

    def forward(self,
                self_state_trace: torch.Tensor,
                observed_state_trace: torch.Tensor,
                self_activated_state: torch.Tensor,
                social_context: Optional[torch.Tensor] = None,
                uncertainty: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            self_state_trace: The agent's own neural activation history (B, D, M)
            observed_state_trace: The observed agent's neural history (B, D, M)
            self_activated_state: The agent's current post-activation state (B, D)
            social_context: Optional social context for oxytocin
            uncertainty: Optional uncertainty for serotonin

        Returns:
            synaptic_modulation: A modulation vector for the CTM's internal state. (B, D)
            empathy_reward: A reward signal based on successful mirroring. (B, 1)

        """
        batch_size, d_model, mem_len = self_state_trace.shape
        device = self_state_trace.device

        # 1. Compute Neural Resonance
        # Compare the traces of each neuron between self and other
        # Lower difference means higher similarity/resonance.
        trace_diff = torch.abs(self_state_trace - observed_state_trace) # (B, D, M)
        
        # Pass the difference through the resonance computer to get a per-neuron score.
        # Input shape (B * D, M) -> output (B * D, 1)
        resonance_scores = self.resonance_computer(trace_diff.view(-1, mem_len))
        resonance_scores = resonance_scores.view(batch_size, d_model) # (B, D)
        
        # Inverse of score -> high resonance for low diff. We want to reward high resonance.
        # Using negative score is also an option. Here, inverse is used.
        resonance_gate = 1.0 / (resonance_scores + 1e-6)
        resonance_gate = torch.clamp(resonance_gate, 0, 5.0).detach() # Detach to act as a gate

        # 2. Mirror Observed State via Attention
        # The agent attends to the observed agent's state to mirror it.
        # Query: self_activated_state
        # Key/Value: observed_state (using the most recent activation from trace)
        observed_current_state = observed_state_trace[:, :, -1] # (B, D)

        mirrored_state, _ = self.mirroring_attention(
            query=self_activated_state.unsqueeze(1),
            key=observed_current_state.unsqueeze(1),
            value=observed_current_state.unsqueeze(1)
        )
        mirrored_state = mirrored_state.squeeze(1) # (B, D)

        # Apply neuromodulators to mirrored_state
        modulation = torch.ones_like(mirrored_state)
        if social_context is not None:
            modulation *= self.neuromodulators['oxytocin'](mirrored_state, social_context)
        if uncertainty is not None:
            modulation *= self.neuromodulators['serotonin'](mirrored_state, uncertainty)
        mirrored_state = mirrored_state * modulation

        # 3. Generate Synaptic Modulation
        # The modulation is guided by the mirrored state and gated by resonance
        synaptic_modulation = self.synaptic_modulator(mirrored_state)
        gated_modulation = synaptic_modulation * resonance_gate

        # 4. Generate Reward
        # Reward is proportional to the overall resonance, weighted by the modulation
        reward_input = (gated_modulation * resonance_gate).detach() # Reward based on resonant action
        empathy_reward = self.reward_generator(reward_input).mean(dim=1)

        return gated_modulation, empathy_reward

class MirrorNeuronLayer(nn.Module):
    """
    Implements a high-level, cognitive form of empathy to encourage selfless behavior.

    This module models the function of mirror neurons by rewarding the agent for
    actions that are predicted to be beneficial to another agent. It uses helper
    modules (`EmotionStateTracker`, `GoalPredictor`) to infer the emotional state
    and goals of an observed agent from its neural activity. It then generates a
    "selfless reward" if the agent's actions are likely to assist with the
    observed agent's goals or alleviate a perceived negative emotional state.
    The module maintains its own internal state across interactions using
    registered buffers for emotion and goal tracking.

    Args:
        d_model (int): The core dimensionality of the CTM's latent space.
        num_heads (int): The number of attention heads for the empathy computation.
        dropout (float): The dropout rate for the attention mechanism.
        num_emotion_dim (int): The dimensionality of the emotion state vectors.
        goal_dim (int): The dimensionality of the predicted goal vectors.
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1,
                 num_emotion_dim: int = 4, goal_dim: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_emotion_dim = num_emotion_dim
        self.goal_dim = goal_dim

        self.emotion_projection = nn.Linear(num_emotion_dim, d_model)
        
        # Emotion state trackers
        self.self_emotion_tracker = EmotionStateTracker(d_model, num_emotion_dim)
        self.observed_emotion_tracker = EmotionStateTracker(d_model, num_emotion_dim)
        
        # Goal predictor
        self.goal_predictor = GoalPredictor(d_model, goal_dim)
        
        # Empathy computation
        self.empathy_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Assistance generator
        self.assistance_net = nn.Sequential(
            nn.Linear(goal_dim + num_emotion_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()  # Assistance probability
        )
        
        # Reward generator
        self.reward_net = nn.Sequential(
            nn.Linear(goal_dim + num_emotion_dim, num_emotion_dim),
            nn.ReLU(),
            nn.Linear(num_emotion_dim, 1),
            nn.Tanh()
        )
        
        # Action-outcome association
        self.action_association = nn.GRU(d_model, d_model, batch_first=True)
        
        # Modulation and normalization
        self.modulation_net = nn.Sequential(
            nn.Linear(d_model + goal_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Tanh()
        )
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Emotional regulation
        self.regulation_gru = nn.GRU(num_emotion_dim, num_emotion_dim, num_layers=1, batch_first=True)
        # Integrate neuromodulators for enhanced mirror neuron processing
        self.neuromodulators = nn.ModuleDict({
            'oxytocin': OxytocinModulator(d_model),
            'dopamine': DopamineModulator(d_model)
        })

    def regulate_emotion(self, emotion, num_steps=2):
        h = torch.zeros(1, emotion.size(0), self.num_emotion_dim, device=emotion.device)
        input_seq = emotion.unsqueeze(1).repeat(1, num_steps, 1)
        regulated, _ = self.regulation_gru(input_seq, h)
        return regulated[:, -1, :]

    def forward(self, self_state: torch.Tensor, observed_state: torch.Tensor,
                prev_self_emotion: torch.Tensor, prev_observed_emotion: torch.Tensor,
                prev_observed_goal: torch.Tensor,
                social_context: Optional[torch.Tensor] = None,
                reward_error: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            self_state: Current agent's neural state [batch, seq_len, d_model]
            observed_state: Observed agent's neural state [batch, seq_len, d_model]
            prev_self_emotion: Previous self emotion state [batch, seq_len, num_emotion_dim]
            prev_observed_emotion: Previous observed emotion [batch, seq_len, num_emotion_dim]
            prev_observed_goal: Previous observed goal [batch, seq_len, goal_dim]
            social_context: Optional social context for oxytocin
            reward_error: Optional reward error for dopamine
            
        Returns:
            modulated_state: Modulated neural state [batch, seq_len, d_model]
            current_self_emotion: Updated self emotion state [batch, seq_len, num_emotion_dim]
            current_observed_goal: Updated observed goal [batch, seq_len, goal_dim]
            reward: Computed reward signal [batch, seq_len, 1]
        """
        # Track emotion states
        current_self_emotion = self.self_emotion_tracker(self_state, prev_self_emotion)
        current_self_emotion = self.regulate_emotion(current_self_emotion)
        current_observed_emotion = self.observed_emotion_tracker(observed_state, prev_observed_emotion)
        current_observed_emotion = self.regulate_emotion(current_observed_emotion)
        
        # Predict internal goal of observed agent
        current_observed_goal = self.goal_predictor(observed_state, prev_observed_goal)
        
        # Compute empathy based on emotion similarity
        empathy, _ = self.empathy_attention(
            query=self.emotion_projection(current_self_emotion),
            key=self.emotion_projection(current_observed_emotion),
            value=self.emotion_projection(current_observed_emotion)
        )
        
        # Apply neuromodulators to empathy
        modulation = torch.ones_like(empathy)
        if social_context is not None:
            modulation *= self.neuromodulators['oxytocin'](empathy, social_context)
        if reward_error is not None:
            modulation *= self.neuromodulators['dopamine'](empathy, reward_error)
        empathy = empathy * modulation
        
        # Generate assistance signal based on predicted goal and emotion
        goal_emotion = torch.cat([current_observed_goal, current_observed_emotion], dim=-1)
        assistance = self.assistance_net(goal_emotion)
        
        # Generate reward signal based on goal progress
        goal_progress = torch.norm(current_observed_goal - prev_observed_goal, dim=-1, keepdim=True)
        reward = self.reward_net(goal_emotion) * goal_progress
        
        # Associate actions with positive outcomes
        reward_assisted = reward * assistance
        reward_expanded = reward_assisted.expand(-1, -1, self.d_model)
        associated_state, _ = self.action_association(self_state, reward_expanded.permute(1,0,2))
        
        # Generate modulation signal using goal context
        modulated = self.modulation_net(torch.cat([associated_state, current_observed_goal], dim=-1))
        
        # Apply modulation with residual connection
        output_state = self.layer_norm(self_state + modulated)
        
        return output_state, current_self_emotion, current_observed_goal, reward
        
    @torch.no_grad()
    def _update_jepa_target_encoder(self):
        """
        Performs momentum update of the JEPA target patch encoder parameters
        using the online patch encoder (self.dynamic_entropy_patcher).
        This should be called by the training loop after optimizer.step().
        """
        if not self.config.use_jepa_training or \
           self.dynamic_entropy_patcher is None or self.jepa_target_patch_encoder is None:
            # Only update if JEPA is active and both encoders exist.
            # No self.training check here, as it might be called during eval for consistency if needed,
            # though typically only during training.
            return
        
        m = self.config.jepa_momentum_beta
        for param_online, param_target in zip(self.dynamic_entropy_patcher.parameters(), self.jepa_target_patch_encoder.parameters()):
            param_target.data.mul_(m).add_((1 - m) * param_online.data)
            
    def start_realtime_voice_streaming(self, duration=10):
        """
        Starts the real-time voice streaming session.
        """
        from .realtime_voice_module import RealtimeVoiceStreamer
        streamer = RealtimeVoiceStreamer(self, self.config)
        streamer.run(duration=duration)
        
    
    def generate_text_and_audio_simultaneously(self, text_prompt: str,
                                               audio_duration_seconds: float = 2.0,
                                               sample_rate: int = 16000,
                                               num_inference_steps: int = 50,
                                               generator: Optional[torch.Generator] = None) -> Tuple[str, torch.Tensor]:
        """
        Generates text and audio simultaneously using a unified binary representation.

        Args:
            text_prompt (str): The text prompt for generation.
            audio_duration_seconds (float): Desired duration of the audio output.
            sample_rate (int): Sample rate for the audio.
            num_inference_steps (int): Number of steps for the diffusion sampler.
            generator (Optional[torch.Generator]): PyTorch generator for reproducibility.

        Returns:
            Tuple[str, torch.Tensor]: A tuple containing the generated text and the generated audio tensor.
        """
        device = self.device_container.device
        
        # 1. Prepare text input
        text_bytes = torch.tensor(list(text_prompt.encode('utf-8')), dtype=torch.uint8, device=device)
        
        # 2. Prepare audio template (zeros)
        num_audio_samples = int(audio_duration_seconds * sample_rate)
        audio_template_numeric = torch.zeros(1, num_audio_samples, device=device) # Batch of 1
        
        # Convert audio template to bytes. Assuming float32 for audio samples.
        audio_template_bytes = batched_numeric_tensor_to_bytes(audio_template_numeric, source_dtype=np.float32).squeeze(0)

        # 3. Create combined byte sequence for CTM input
        separator = torch.tensor([255, 0, 255, 0, 255, 0, 255, 0], dtype=torch.uint8, device=device) # A unique separator
        
        combined_input_bytes = torch.cat([text_bytes, separator, audio_template_bytes]).unsqueeze(0) # Add batch dim

        # 4. Prepare features for the diffusion model
        kv_features, _, _, _ = self._prepare_input_features(combined_input_bytes)
        gen_shape = kv_features.shape

        # 5. Run the fast sampling process
        generated_output, sampling_info = self.ultra_fast_integration_flow_generation(
            shape=gen_shape,
            initial_byte_sequence_for_inference=combined_input_bytes
        )

        generated_bytes = generated_output.squeeze(0)
        
        # 6. Decode the output
        # Find separator in the generated bytes
        separator_np = separator.cpu().numpy()
        generated_np = generated_bytes.cpu().numpy()
        
        sep_idx = -1
        # Simple sliding window search for the separator
        for i in range(len(generated_np) - len(separator_np) + 1):
            if np.array_equal(generated_np[i:i+len(separator_np)], separator_np):
                sep_idx = i
                break

        if sep_idx == -1:
            # Separator not found; treat as text
            text_result = generated_bytes.cpu().numpy().tobytes().decode('utf-8', errors='ignore')
            audio_result = torch.zeros(1, num_audio_samples, device=device)
        else:
            text_part_bytes = generated_bytes[:sep_idx]
            audio_part_bytes = generated_bytes[sep_idx+len(separator_np):]

            text_result = text_part_bytes.cpu().numpy().tobytes().decode('utf-8', errors='ignore')
            
            # Convert audio bytes back to numeric tensor
            audio_part_tensor_uint8 = audio_part_bytes.unsqueeze(0) # Add batch dim
            # Ensure the length of audio bytes is a multiple of item_size (4 for float32)
            item_size = 4
            if audio_part_tensor_uint8.shape[1] % item_size != 0:
                pad_size = item_size - (audio_part_tensor_uint8.shape[1] % item_size)
                padding = torch.zeros((1, pad_size), dtype=torch.uint8, device=device)
                audio_part_tensor_uint8 = torch.cat([audio_part_tensor_uint8, padding], dim=1)

            try:
                audio_result = batched_bytes_to_numeric_tensor(audio_part_tensor_uint8, item_size=item_size, target_dtype=np.float32)
            except ValueError as e:
                # Audio conversion error
                audio_result = torch.zeros(1, num_audio_samples, device=device)

        return text_result, audio_result.squeeze(0)
        
class FrequencyDomainAwareAttention(nn.Module):
    """Generalized HiPA that works across different modalities with intelligent task detection."""

    def __init__(self, embed_dim=512, num_heads=8, task_analyzer: 'TaskAnalyzer' = None,
                 config: Optional[EnhancedCTMConfig] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.task_analyzer = task_analyzer
        self.config = config

        # Multi-head attention for frequency-aware processing
        self.freq_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        # Frequency enhancement networks
        self.freq_enhancer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x: torch.Tensor, hipa_control_signal: Optional[torch.Tensor] = None, context_hints: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with dynamically controlled frequency enhancement.
        HiPA is now primarily controlled by hipa_control_signal.
        Task_analyzer might still provide base modality characteristics.
        """
        batch_size, seq_len, embed_dim = x.shape

        modality_config = self.task_analyzer.detect_modality(x, task_id=None, context_hints=context_hints)

        x_processed = x
       
        # This implies freq_attention is a standard MHA. The "frequency aware" part was in apply_frequency_enhancement.
        # So, we apply MHA on x_processed.

        # Reshape for attention if needed (original code had this before freq_attention)
        original_shape = x_processed.shape
        if len(x_processed.shape) > 3:
            x_flat = x_processed.view(batch_size, -1, x_processed.shape[-1])
        else:
            x_flat = x_processed

         # Standard attention mechanism (using self.freq_attention as the MHA layer)
        if x_flat.shape[-1] == self.embed_dim:
            attn_output, _ = self.freq_attention(x_flat, x_flat, x_flat)
            # The original code also had a self.freq_enhancer after attention.
            # If this is part of HIPA, it should also be controlled or always applied after HIPA-MHA.
            # Let's assume it's part of the HIPA block.
            final_output_flat = self.freq_enhancer(attn_output)
        else:
            # If dimensions don't match, or if HIPA was fully bypassed and no attention is desired here.
            # This path needs clarification. For now, assume if HIPA is on, attention runs.
            # If HIPA was off (x_processed is x), does it still go through this attention?
            # Let's assume this MHA is always run on x_processed.
             # If x_flat.shape[-1] != self.embed_dim, it's an issue.
             raise ValueError(f"Dimension mismatch for attention: x_flat has dim {x_flat.shape[-1]}, expected {self.embed_dim}")

        if len(original_shape) > 3:
            final_output = final_output_flat.view(original_shape[0], *original_shape[1:-1], self.embed_dim)
        else:
            final_output = final_output_flat
            
        return final_output, modality_config


class TemporalSpatialTracker(nn.Module):
    """
    Temporal-Spatial Awareness Tracker for preserving timestamp and spatial data
    during diffusion processes. This component tracks both temporal sequences and
    spatial relationships to maintain data integrity across diffusion steps.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # Temporal encoding components
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, config.d_model // 4),  # Timestamp input
            nn.SiLU(),
            nn.Linear(config.d_model // 4, config.d_model // 2),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, config.d_model)
        )
        
        # Spatial relationship encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # Temporal-spatial fusion attention
        self.temporal_spatial_attention = nn.MultiheadAttention(
            config.d_model, num_heads=8, batch_first=True
        )
        
        # Positional encoding for sequences
        self.positional_encoding = nn.Parameter(
            torch.randn(config.max_sequence_length, config.d_model) * 0.02
        )
        
        # Timestamp preservation network
        self.timestamp_preservers = nn.ModuleDict({
            'linear': nn.Linear(config.d_model, config.d_model),
            'attention': nn.MultiheadAttention(config.d_model, num_heads=4, batch_first=True),
            'temporal_conv': nn.Conv1d(config.d_model, config.d_model, kernel_size=3, padding=1)
        })
        
        # Spatial context preservation
        self.spatial_preservers = nn.ModuleDict({
            'local': nn.Conv1d(config.d_model, config.d_model, kernel_size=3, padding=1),
            'global': nn.Linear(config.d_model, config.d_model),
            'hierarchical': nn.ModuleList([
                nn.Conv1d(config.d_model, config.d_model, kernel_size=k, padding=k//2)
                for k in [3, 5, 7, 9]  # Multi-scale spatial awareness
            ])
        })
        
        # Diffusion step aware processing
        self.step_aware_processor = nn.Sequential(
            nn.Linear(config.d_model + 1, config.d_model),  # +1 for diffusion step
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
    def encode_timestamps(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamp information for preservation during diffusion."""
        # Normalize timestamps to [0, 1] range
        if timestamps.max() > 1.0:
            timestamps = timestamps / timestamps.max()
        
        # Encode temporal information
        temporal_features = self.temporal_encoder(timestamps.unsqueeze(-1))
        return temporal_features
    
    def preserve_spatial_relationships(self, x: torch.Tensor, diffusion_step: int) -> torch.Tensor:
        """Preserve spatial relationships during diffusion steps."""
        batch_size, seq_len, d_model = x.shape
        
        # Apply multi-scale spatial preservation
        spatial_features = []
        
        # Local spatial relationships
        x_transposed = x.transpose(1, 2)  # (B, d_model, seq_len)
        local_spatial = self.spatial_preservers['local'](x_transposed).transpose(1, 2)
        spatial_features.append(local_spatial)
        
        # Global spatial context
        global_spatial = self.spatial_preservers['global'](x)
        spatial_features.append(global_spatial)
        
        # Hierarchical spatial processing
        for conv_layer in self.spatial_preservers['hierarchical']:
            hierarchical_spatial = conv_layer(x_transposed).transpose(1, 2)
            spatial_features.append(hierarchical_spatial)
        
        # Combine spatial features with attention
        combined_spatial = torch.stack(spatial_features, dim=1)  # (B, num_scales, seq_len, d_model)
        combined_spatial = combined_spatial.view(batch_size, -1, d_model)
        
        # Apply spatial attention
        spatial_attended, _ = self.temporal_spatial_attention(
            combined_spatial, combined_spatial, combined_spatial
        )
        
        # Average across scales
        spatial_preserved = spatial_attended.view(batch_size, len(spatial_features), seq_len, d_model).mean(dim=1)
        
        return spatial_preserved
    
    def apply_diffusion_step_awareness(self, x: torch.Tensor, diffusion_step: int,
                                     total_steps: int) -> torch.Tensor:
        """Apply diffusion step awareness to preserve temporal-spatial information."""
        batch_size, seq_len, d_model = x.shape
        
        # Normalize diffusion step
        step_ratio = torch.tensor(diffusion_step / total_steps, device=x.device, dtype=x.dtype)
        step_ratio = step_ratio.expand(batch_size, seq_len, 1)
        
        # Combine with features
        x_with_step = torch.cat([x, step_ratio], dim=-1)
        
        # Process with step awareness
        step_aware_features = self.step_aware_processor(x_with_step)
        
        return step_aware_features
    
    def forward(self, x: torch.Tensor, timestamps: torch.Tensor = None,
                diffusion_step: int = 0, total_steps: int = 1000) -> torch.Tensor:
        """
        Forward pass with full temporal-spatial awareness.
        
        Args:
            x: Input features (B, seq_len, d_model)
            timestamps: Timestamp information (B, seq_len) or (B,)
            diffusion_step: Current diffusion step
            total_steps: Total diffusion steps
            
        Returns:
            Enhanced features with preserved temporal-spatial information
        """
        batch_size, seq_len, d_model = x.shape
        
        # Add positional encoding for sequence awareness
        if seq_len <= self.positional_encoding.size(0):
            pos_encoding = self.positional_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
            x = x + pos_encoding
        
        # Encode timestamps if provided
        if timestamps is not None:
            if timestamps.dim() == 1:
                timestamps = timestamps.unsqueeze(1).expand(-1, seq_len)
            temporal_features = self.encode_timestamps(timestamps)
            x = x + temporal_features
        
        # Preserve spatial relationships
        spatial_preserved = self.preserve_spatial_relationships(x, diffusion_step)
        
        # Apply diffusion step awareness
        step_aware_features = self.apply_diffusion_step_awareness(
            spatial_preserved, diffusion_step, total_steps
        )
        
        # Final temporal-spatial fusion
        enhanced_features, _ = self.temporal_spatial_attention(
            step_aware_features, step_aware_features, step_aware_features
        )
        
        return enhanced_features


# ============================================================================
# CTM INTEGRATION FLOW TRAINER WITH GPU OPTIMIZATIONS
# ============================================================================

class CTMIntegrationFlowTrainer:
    """
    Specialized trainer for CTM-guided Integration Flow that optimizes
    the CTM's guidance capability for fast, high-quality generation.
    Includes GPU optimizations and memory management.
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Training-specific parameters
        self.guidance_loss_weight = 0.3
        self.consistency_loss_weight = 0.2
        self.quality_loss_weight = 0.5
        
        # GPU optimization settings
        self.enable_mixed_precision = True
        self.gradient_accumulation_steps = 4
        self.max_grad_norm = 1.0
        
        # Memory optimization
        self.enable_gradient_checkpointing = True
        self.batch_size_optimization = True
        
    def compute_guidance_loss(self, generated, target, ctm_data):
        """
        Loss that encourages CTM to provide better guidance for Integration Flow.
        """
        # Main reconstruction loss
        recon_loss = F.mse_loss(generated, target)
        
        # CTM guidance consistency loss
        consistency_loss = 0.0
        if 'certainties' in ctm_data:
            # Encourage high certainty when generation is good
            certainty = ctm_data['certainties'][:, 0, -1]  # (B,)
            generation_quality = -F.mse_loss(generated, target, reduction='none').mean(dim=tuple(range(1, len(generated.shape))))
            
            # High quality should correlate with high certainty
            consistency_loss = F.mse_loss(certainty, torch.sigmoid(generation_quality))
        
        # Total loss
        total_loss = (self.quality_loss_weight * recon_loss +
                     self.consistency_loss_weight * consistency_loss)
        
        return total_loss, {
            'recon_loss': recon_loss.item(),
            'consistency_loss': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
            'total_loss': total_loss.item()
        }
    
    def train_step(self, batch_data, optimizer, scaler=None):
        """Single training step optimized for CTM-guided Integration Flow with GPU optimizations"""
        
        # Get clean data
        clean_data = batch_data['clean']
        
        # Add noise for Integration Flow training
        noise = torch.randn_like(clean_data)
        
        # GPU optimization: Use mixed precision if available
        if self.enable_mixed_precision and scaler is not None:
            with torch.amp.autocast("cuda",):
                # Get CTM context
                ctm_features = self.model.compute_features(clean_data)
                ctm_data = self.model.ctm_core.forward_with_full_tracking(ctm_features)
                
                # Generate using CTM-guided Integration Flow
                generated = self.model.diffusion.integration_flow_one_step_generation(
                    clean_data.shape, ctm_data, task_id=0
                )
                
                # Compute loss
                loss, metrics = self.compute_guidance_loss(generated, clean_data, ctm_data)
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        else:
            # Standard precision training
            # Get CTM context
            ctm_features = self.model.compute_features(clean_data)
            ctm_data = self.model.ctm_core.forward_with_full_tracking(ctm_features)
            
            # Generate using CTM-guided Integration Flow
            generated = self.model.diffusion.integration_flow_one_step_generation(
                clean_data.shape, ctm_data, task_id=0
            )
            
            # Compute loss
            loss, metrics = self.compute_guidance_loss(generated, clean_data, ctm_data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            
            optimizer.step()
        
        return metrics

class BasalGangliaMechanism(nn.Module):
    """
    Models the function of the basal ganglia, acting as a real-time action gating system.

    This module operates within the CTM's iterative thought process, inspecting
    the developing action plan (`synchronisation_action`) at each step. It uses
    "Go" (direct) and "NoGo" (indirect) pathways to learn to approve or suppress
    actions. It learns implicitly from the global training loss, reinforcing
    thought patterns that lead to low-loss outcomes and inhibiting those that
    lead to high-loss outcomes. It also includes a dopaminergic system that
    learns to predict the reward value of states, contributing to the gating
    decision. Its primary role is to ensure that the thoughts and reasoning
    paths generated by the CTM are coherent and contextually appropriate.

    Args:
        d_model (int): The dimensionality of the CTM's thought vector.
        action_dim (int): The dimensionality of the action representation to be gated.
        dopamine_dim (int): The dimensionality of the internal dopamine prediction network.
    """
    def __init__(self, d_model: int, action_dim: int, dopamine_dim: int = 32, context_dim: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.dopamine_dim = dopamine_dim
        if context_dim is None:
            context_dim = d_model
        self.context_dim = context_dim
        
        # Direct pathway (Go signal)
        self.direct_pathway = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, action_dim),
            nn.Sigmoid()  # Produces a gate between 0 and 1
        )
        
        # Indirect pathway (NoGo signal)
        self.indirect_pathway = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, action_dim),
            nn.Sigmoid()  # Produces an inhibition gate between 0 and 1
        )
        
        # Dopaminergic system (reward prediction error)
        self.dopamine_predictor = nn.Sequential(
            nn.Linear(d_model, dopamine_dim),
            nn.ReLU(),
            nn.Linear(dopamine_dim, 1),
            nn.Tanh() # Predicts a reward value between -1 and 1
        )
        
        # Contextual biasing
        self.context_gating = nn.Sequential(
            nn.Linear(self.context_dim, d_model),
            nn.Sigmoid()
        )
        self.reward_input_proj = nn.Linear(d_model + action_dim, d_model)
        self.dopamine_mod = DopamineModulator(self.dopamine_dim)
        self.serotonin_mod = SerotoninModulator(self.action_dim)
        self.norepi_mod = NorepinephrineModulator(self.context_dim)
        self.gaba_mod = GABAModulator(self.action_dim)
    
    def forward(self, thought_vector: torch.Tensor, context: torch.Tensor,
                reward_signal: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            thought_vector: Current thought vector (B, d_model)
            context: Contextual information (B, d_model)
            reward_signal: Optional external reward signal (B, 1)
            
        Returns:
            net_action_gate: A gating vector to modulate the continuous action. (B, action_dim)
            dopamine_error: Dopamine prediction error signal (B, 1)
        """
        # Apply contextual gating
        context_gate = self.context_gating(context)
        gated_thought = thought_vector * context_gate
        
        # Direct pathway: "Go" signal, encourages action features
        selection_gate = self.direct_pathway(gated_thought)
        
        # Indirect pathway: "NoGo" signal, inhibits action features
        inhibition_gate = self.indirect_pathway(gated_thought)
        
        # Combine pathways: action features are passed if Go is high and NoGo is low
        net_action_gate = selection_gate * (1 - inhibition_gate)
        
        # Dopamine reward prediction
        predicted_reward = self.dopamine_predictor(gated_thought)
        
        # Calculate dopamine error if reward signal is provided
        dopamine_error = -predicted_reward # By default, the error is the negative predicted reward
        if reward_signal is not None:
            dopamine_error = reward_signal - predicted_reward
        
        # Apply neuromodulators as enhancements
        activity_level = gated_thought.norm(dim=-1)
        novelty = (thought_vector - context).norm(dim=-1)
        uncertainty = activity_level  # Using activity as proxy for uncertainty
        
        predicted_reward = self.dopamine_mod(predicted_reward, dopamine_error)
        selection_gate = self.dopamine_mod(selection_gate, dopamine_error)
        inhibition_gate = self.serotonin_mod(inhibition_gate, uncertainty)
        inhibition_gate = self.gaba_mod(inhibition_gate, activity_level)
        context_gate = self.norepi_mod(context_gate, novelty)
        
        # Recompute net_action_gate with modulated values
        net_action_gate = selection_gate * (1 - inhibition_gate)
        
        return net_action_gate, dopamine_error

    def select_action(self, action_candidates: List[torch.Tensor],
                     thought_vector: torch.Tensor,
                     context: torch.Tensor) -> torch.Tensor:
        """
        Selects the best action from candidates based on predicted reward.
        
        Args:
            action_candidates: List of candidate action tensors (each B x action_dim)
            thought_vector: Current thought vector (B x d_model)
            context: Contextual information (B x d_model)
            
        Returns:
            Selected action tensor (B x action_dim)
        """
        predicted_rewards = []
        for candidate in action_candidates:
            # Simulate the gated action
            gate, _ = self.forward(thought_vector, context)
            gated_candidate = candidate * gate
            
            # Predict reward for this gated action
            combined = torch.cat([thought_vector, gated_candidate], dim=-1)
            projected_input = self.reward_input_proj(combined)
            pred_reward = self.dopamine_predictor(projected_input).squeeze(-1)
            predicted_rewards.append(pred_reward)
        
        # Stack rewards and select best
        reward_stack = torch.stack(predicted_rewards, dim=1)  # (B, num_candidates)
        best_indices = reward_stack.argmax(dim=1)  # (B,)
        
        # Select the best actions
        selected_actions = torch.stack([
            action_candidates[idx][batch_idx]
            for batch_idx, idx in enumerate(best_indices)
        ], dim=0)
        
        return selected_actions

class JEPAPredictor(nn.Module):
    """
    MLP predictor for JEPA.
    Predicts target patch embedding(s) from context patch embedding(s).
    Input dimension should be the patch_embedding_dim from LearnedBytePatcherEncoder.
    Output dimension will be patch_embedding_dim * num_target_blocks.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int): # output_dim here is patch_embedding_dim * num_target_blocks
        super().__init__()
        # Ensure hidden_dim is reasonable
        # The output_dim passed here is already patch_embedding_dim * num_target_blocks
        actual_hidden_dim = max(hidden_dim, input_dim // 2, output_dim // 4, 64) # Adjusted output_dim factor for hidden layer

        self.network = nn.Sequential(
            nn.Linear(input_dim, actual_hidden_dim),
            nn.LayerNorm(actual_hidden_dim),
            nn.GELU(),
            nn.Linear(actual_hidden_dim, actual_hidden_dim),
            nn.LayerNorm(actual_hidden_dim),
            nn.GELU(),
            nn.Linear(actual_hidden_dim, output_dim) # This output_dim is patch_embedding_dim * num_target_blocks
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be the representation of context patches, e.g., (B, D_embed)
        # Output will be (B, patch_embedding_dim * num_target_blocks)
        return self.network(x)

"""
Hierarchical Reasoning Continuous Thought Machine (HR-CTM)
This model integrates the principles of the Hierarchical Reasoning Model (HRM)
into the Continuous Thought Machine (CTM) architecture. It features a two-level
recurrent system to achieve greater computational depth and reasoning capability.
Key Features:
1.  **Hierarchical Structure**: A high-level, slow-updating module (H-module) for
    abstract planning and a low-level, fast-updating module (L-module) for
    detailed computation.
2.  **Hierarchical Convergence**: The L-module performs multiple computational steps
    to reach a local equilibrium before the H-module performs a single update,
    enabling deep, nested reasoning.
3.  **Preservation of CTM Principles**: Leverages the core CTM concepts like
    synchronization-as-representation and neuron-level models within the
    hierarchical framework.
4.  **Replacement of Frequency Boosts**: The explicit frequency-based creative boosts
    are replaced by the intrinsic multi-timescale dynamics of the H and L modules.
"""

class HRM_H_Module(nn.Module):
    """The High-Level, slow-updating recurrent module for the HR-CTM."""
    def __init__(self, config: EnhancedCTMConfig):
        super().__init__()
        self.config = config
        self.base_thresholds = {'critical': 0.99, 'medium': 0.8, 'low': 0.5}
        self.confidence_thresholds = {k: 0.0 for k in self.base_thresholds}  # Start at 0
        self.initial_epochs = 5  # Epochs before ramp-up starts
        self.current_epoch = 0
        # This module integrates the result from the L-module (zL) into its own state (zH).
        self.mamba = Mamba2Block(d_model=config.d_model)
        self.sparse_attn = WINAAttention(d_model=config.d_model, n_heads=config.n_heads, dropout=config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.planning_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.ReLU(),
            nn.Linear(config.d_model * 2, config.d_model * 4),
            nn.ReLU(),
            nn.Linear(config.d_model * 4, config.d_model * 2),
            nn.ReLU(),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model)
        )
        self.norm3 = nn.LayerNorm(config.d_model)
        # Project zL to match d_model for attention
        self.zl_proj = nn.Linear(config.d_model, config.d_model)  # Assuming zL has d_model
        patcher_config = {
            'embedding_dim': config.patch_embedding_dim,
            'patch_cnn_channels': config.patch_encoder_cnn_channels,
            'patching_mode': config.entropy_patcher_threshold_type,
            'global_threshold': config.entropy_patcher_global_threshold,
            'relative_threshold': config.entropy_patcher_relative_threshold,
            'min_patch_size': config.entropy_patcher_min_patch_size,
            'max_patch_size': config.entropy_patcher_max_patch_size,
            'entropy_byte_vocab_size': config.entropy_model_byte_vocab_size,
            'entropy_embedding_dim': config.entropy_model_embedding_dim,
            'entropy_hidden_dim': config.entropy_model_hidden_dim,
            'entropy_num_layers': config.entropy_model_num_layers,
            'entropy_dropout': config.entropy_model_dropout
        }
        if getattr(config, 'use_program_synthesizer', False):
            self.program_synthesizer = ProgramSynthesizer(
                d_model=config.d_model,
                n_heads=config.program_synth_n_heads,
                n_layers=config.program_synth_n_layers,
                d_ff=config.program_synth_d_ff,
                dropout=config.dropout,
                max_gen_len=config.max_sequence_length, # Or a more specific config
                patcher_config=patcher_config
            )
        else:
            self.program_synthesizer = None
        self.hypernet = HyperNetwork(config.d_model * 2, config.d_model)
        self.meta_learner = nn.Linear(config.d_model * 2, config.d_model)  # Base learner, params generated by hypernet
        self.foresight = ForesightSimulator(config.d_model)
        self.max_recursion = config.max_recursion
        self.early_stop_threshold = config.early_stop_threshold
        self.program_feedback_proj = nn.Linear(config.d_model, config.d_model)
        self.shared_neuromodulator_manager = NeuromodulatorManager(config)
        self.thought_ctm = OriginalCTMCore(config, neuromodulator_manager=self.shared_neuromodulator_manager)
        self.thought_feedback_proj = nn.Linear(config.ctm_out_dims, config.d_model)
        
        # Add CTM-like components for H-module
        # Using N=1 since it's used as a regular MLP, not per-neuron.
        self.h_synapses = SuperLinear(2, 1, N=config.d_model, depth=config.ctm_synapse_depth, dropout=config.ctm_dropout)
        self.h_trace_processor = SuperLinear(config.ctm_memory_length, 1, N=config.d_model, depth=config.ctm_deep_nlms, dropout=config.ctm_dropout)
        self.h_q_proj = nn.Linear(config.d_model, config.d_model)  # For H-module sync-based query
        
    def forward(self, zH: torch.Tensor, zL: torch.Tensor, retrieved_memory: torch.Tensor, thought_guidance: bool = True, confidence_level: str = 'medium') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            zH (torch.Tensor): Current high-level state.
            zL (torch.Tensor): Final low-level state from the L-cycle.
            retrieved_memory (torch.Tensor): Memory retrieved from the LTM.
            thought_guidance (bool): Flag to switch to direct CTM thought vector guidance. #Recommended on for most model usage.
            confidence_level (str): 'critical', 'medium', or 'low'
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Next high-level state, encoded_patches (or None), patch_indices (or None), entropy_aux_loss (or 0), confidence.
        """
        current_zH = zH
        prev_zH = None
        depth = 0
        encoded_patches = None
        patch_indices = None
        entropy_aux_loss = torch.tensor(0.0, device=zH.device)
        deltas = []
        
        # Initialize H-module trace
        h_trace = torch.zeros_like(current_zH.unsqueeze(-1).repeat(1, 1, self.config.ctm_memory_length))
        
        # Initialize sync for H-module
        decay_alpha_h, decay_beta_h = None, None
        r_h = torch.exp(torch.tensor(-0.1))  # Example decay rate
        
        while depth < self.max_recursion:
            # Compute H-module synchronization (pulsing)
            sync_h, decay_alpha_h, decay_beta_h = self.compute_synchronisation(
                current_zH, decay_alpha_h, decay_beta_h, r_h, 'action'  # Reuse 'action' type
            )
            
            # The query is the current high-level state modulated by sync
            q = self.h_q_proj(sync_h).unsqueeze(1)
            
            # The key/value is the information from the completed low-level cycle and retrieved memory
            # The retrieved_memory is now a single contextualized vector from the LTM's attention mechanism
            kv = self.zl_proj(zL) + retrieved_memory.squeeze(0) # Squeeze to remove the batch dim of 1
            # Attention step
            kv = (self.zl_proj(zL) + retrieved_memory.squeeze(0)).unsqueeze(1)
            current_zH = current_zH.unsqueeze(1)
            q = self.h_q_proj(sync_h).unsqueeze(1)
            
            # Dynamic routing: Compute WINA scores to decide per token
            scores = self.sparse_attn.wina_sparsifier.compute_wina_scores(current_zH, self.sparse_attn.q_proj.weight)
            route_to_attention = (scores > 0.5).float()  # Example threshold; make learnable
            
            # Mamba path (default/efficient)
            mamba_out = self.mamba(current_zH, confidence_level=confidence_level)
            
            # Sparse attention path (selective)
            attn_out = self.sparse_attn(current_zH, kv, kv)
            
            # Fuse based on routing
            current_zH = current_zH + (mamba_out * (1 - route_to_attention) + attn_out * route_to_attention)
            
            # Repeat for second block (or loop for more)
            scores = self.sparse_attn.wina_sparsifier.compute_wina_scores(current_zH, self.sparse_attn.q_proj.weight)
            route_to_attention = (scores > 0.5).float()
            
            mamba_out = self.mamba(current_zH, confidence_level=confidence_level)
            attn_out = self.sparse_attn(current_zH, kv, kv)
            current_zH = current_zH + (mamba_out * (1 - route_to_attention) + attn_out * route_to_attention)
            
            current_zH = current_zH.squeeze(1)
            
            meta_input = torch.cat([current_zH, zL], dim=-1)
            # Dynamic meta-learning with hypernetwork
            weight, bias = self.hypernet(meta_input)
            meta_update = F.linear(meta_input, weight, bias)
            current_zH = current_zH + meta_update * 0.1  # Small meta-update step
            current_zH = self.norm1(current_zH)
            
            # Additional planning layer
            planning_output = self.planning_mlp(current_zH)
            current_zH = self.norm3(current_zH + planning_output)
            
            # Add foresight simulation
            foresight_adjust = self.foresight(current_zH)
            current_zH = current_zH + foresight_adjust * 0.05
            
            # Add CTM-like synapse and NLM processing
            h_pre_synapse = torch.cat([current_zH, retrieved_memory.squeeze(0)], dim=-1)
            h_state = self.h_synapses(h_pre_synapse.view(h_pre_synapse.shape[0], self.config.d_model, 2))
            h_trace = torch.cat((h_trace[:, :, 1:], h_state.unsqueeze(-1)), dim=-1)
            current_zH = self.h_trace_processor(h_trace)
            
            if not thought_guidance and self.program_synthesizer is not None:
                # Synthesize a program using the new synthesizer
                encoded_patches, patch_indices, entropy_aux_loss = self.program_synthesizer(current_zH)
                
                # Feedback from synthesized program to high-level state
                if encoded_patches is not None and encoded_patches.size(1) > 0:
                    program_feedback = self.program_feedback_proj(encoded_patches.mean(dim=1))
                    current_zH = current_zH + program_feedback * 0.1
            elif not thought_guidance:
                # Program synthesizer is disabled, so we return empty tensors for compatibility
                encoded_patches = None
                patch_indices = None
                entropy_aux_loss = torch.tensor(0.0, device=zH.device)
            quantize = (self.config.quant_enabled_training and self.training) or \
                       (self.config.quant_enabled_inference and not self.training)
            if quantize and self.config.ctm_adaptive_quantization and self.thought_ctm.bitwidth_adapter:
                with torch.no_grad():
                    task_embedding = current_zH.mean(dim=0)
                    bits = self.thought_ctm.bitwidth_adapter(task_embedding)
                
                if quantize and self.config.ctm_quant_policy_search and self.thought_ctm.quant_policy_net:
                    with torch.no_grad():
                        policy_params = self.thought_ctm.quant_policy_net(task_embedding)
                        scale = policy_params[:, 1].mean()
                        zero_point = policy_params[:, 2].mean()
                    q_zH, _, _ = quantize_adaptive(current_zH, bits)
                    current_zH = dequantize_adaptive(q_zH, scale, zero_point)
                else:
                    q_zH, scale, zero_point = quantize_adaptive(current_zH, bits)
                    current_zH = dequantize_adaptive(q_zH, scale, zero_point)

            # Direct CTM thought vector guidance
            ctm_predictions, ctm_certainties, ctm_sync_out = self.thought_ctm(current_zH.unsqueeze(1))
            thought_feedback = self.thought_feedback_proj(ctm_sync_out)
            current_zH = current_zH + thought_feedback * 0.1
            # Set placeholders for return values
            encoded_patches = None
            patch_indices = None
            entropy_aux_loss = torch.tensor(0.0, device=zH.device)
            
            # Early stopping check
            if prev_zH is not None:
                delta = torch.norm(current_zH - prev_zH, dim=-1).mean()
                deltas.append(delta)
                if delta < self.early_stop_threshold:
                    break
            
            prev_zH = current_zH.clone()
            depth += 1
        
        # Hallucination reduction: Compute confidence based on variance of deltas
        if deltas:
            variance = torch.var(torch.stack(deltas))
            confidence = torch.exp(-variance)
            threshold = self.confidence_thresholds.get(confidence_level, 0.8)
            if not self.training and confidence < threshold:
                current_zH = current_zH * 0  # Abstain only during inference
        else:
            confidence = torch.tensor(1.0, device=zH.device)
        
        # The 'program' is now the sequence of encoded patches (or None in direct mode).
        # The other outputs might be used for loss calculation or debugging.
        current_zH = current_zH * self.shared_neuromodulator_manager(current_zH)
        return current_zH, encoded_patches, patch_indices, entropy_aux_loss, confidence
    
    def update_thresholds(self, epoch, total_epochs):
        self.current_epoch = epoch
        if epoch < self.initial_epochs:
            factor = 0.0
        else:
            factor = (epoch - self.initial_epochs) / max(1, total_epochs - self.initial_epochs)
        
        for level in self.confidence_thresholds:
            self.confidence_thresholds[level] = factor * self.base_thresholds[level]

class HRM_L_Module(nn.Module):
    """The Low-Level, fast-updating CTM-based recurrent module for the HR-CTM."""
    def __init__(self, config: EnhancedCTMConfig, parent_ctm: 'HierarchicalCTM'):
        super().__init__()
        self.config = config
        self.d_model = config.ctm_d_model
        self.d_input = config.ctm_input_dim
        
        self.mamba_encoder = Mamba2Block(d_model=self.d_input)
        
        # Inherit synapse and NLM models from parent HierarchicalCTM
        # to ensure they are registered correctly under the main model.
        self.synapses = parent_ctm.synapses
        self.trace_processor = parent_ctm.trace_processor
        
        # Projector for the query, derived from the low-level sync state
        self.q_proj = nn.Linear(parent_ctm.synch_representation_size_action, self.d_input)
        self.top_down_projector = nn.Linear(self.config.d_model, self.d_model)  # Project zH to modulation signal
        
        self.attention = WINAAttention(d_model=self.d_input, n_heads=config.n_heads, dropout=config.dropout)
        if self.config.use_spatial:
            self.spatial_reasoning = SpatialReasoningModule(self.d_model)
            self.three_d_spatial_reasoning = ThreeDSpatialReasoningModule(self.d_model)
        else:
            self.spatial_reasoning = None
            self.three_d_spatial_reasoning = None

    def forward(self,
                activated_zL: torch.Tensor,
                zL_trace: torch.Tensor,
                zH: torch.Tensor,
                x_context: torch.Tensor,
                sync_action: torch.Tensor,
                confidence_level: str = 'medium') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one step of the low-level CTM computation.
    
        Args:
            activated_zL: Current post-activation state of the L-module. (B, D)
            zL_trace: History of pre-activations for the L-module. (B, D, M)
            zH: Current high-level state, provides top-down context. (B, D)
            x_context: External input, provides bottom-up context. (B, S, d_input)
            sync_action: Synchronization representation for generating attention query. (B, sync_dim)
    
        Returns:
            A tuple of (next_activated_zL, next_zL_trace).
        """
        x_context = self.mamba_encoder(x_context, confidence_level=confidence_level).contiguous()
    
        # 1. Interact with context via attention
        # Query is from L-module's own action synchronisation
        q = self.q_proj(sync_action).unsqueeze(1)
        
        # Key/Value is a combination of external input and high-level context
        # Add zH as part of the key/value context
        kv = x_context + zH.unsqueeze(1)
        attn_out, _ = self.attention(q, kv, kv)
        attn_out = attn_out.squeeze(1)
    
        # 2. Form input for synapses
        pre_synapse_input = torch.cat((attn_out, activated_zL), dim=-1)
    
        # 3. Apply Synapses to get pre-activation state
        state = self.synapses(pre_synapse_input)
        top_down_mod = self.top_down_projector(zH)  # (B, D)
        state = state + top_down_mod * 0.3  # Modulate with strength 0.3
        
        # Add parietal-inspired spatial reasoning
        if self.config.use_spatial and self.spatial_reasoning is not None:
            state = self.spatial_reasoning(state.unsqueeze(1)).squeeze(1)
    
        # Add 3D spatial reasoning - assume a 3D grid size, e.g., (4,4,4) if d_model=64
        # Adjust based on actual d_model; here assuming d_model is cube-able
        if self.config.use_spatial and self.three_d_spatial_reasoning is not None:
            cube_root = int(self.d_model ** (1/3))
            grid_3d = (cube_root, cube_root, cube_root)
            state = self.three_d_spatial_reasoning(state.unsqueeze(1), grid_size=grid_3d).squeeze(1)
    
        # 4. Update state trace (memory for NLMs)
        next_zL_trace = torch.cat((zL_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
    
        # 5. Apply Neuron-Level Models (NLMs) to get next post-activation state
        next_activated_zL = self.trace_processor(next_zL_trace)
        
        return next_activated_zL, next_zL_trace
    

  
class HierarchicalCTM(OriginalCTMCore):
    """
    The main Hierarchical Reasoning CTM model.
    Inherits from OriginalCTMCore to reuse helper methods for initialization."
    """
    def __init__(self, config: EnhancedCTMConfig):
        # We call nn.Module's init directly to avoid OriginalCTMCore's full init,
        # as we are building a different structure.
        super().__init__(config)
        self.replay_batch_size = getattr(config, 'replay_batch_size', 4)
        self.d_model = config.ctm_d_model
        self.d_input = config.ctm_input_dim
        self.memory_length = config.ctm_memory_length
        
        # --- Instantiate CTM components needed for the L-module ---
        # These methods are borrowed from OriginalCTMCore
        self.synapses = self.get_synapses(
            config.ctm_synapse_depth, self.d_input + self.d_model, self.d_model, config.ctm_dropout
        )
        self.trace_processor = self.get_neuron_level_models(
            config.ctm_deep_nlms, config.ctm_do_layernorm_nlm, config.ctm_memory_length,
            config.ctm_memory_hidden_dims, self.d_model, config.ctm_dropout_nlm or config.ctm_dropout
        )
        
        # --- Instantiate HRM Modules ---
        self.l_module = HRM_L_Module(config, self)
        self.h_module = HRM_H_Module(config)
        self.ltm = LongTermMemory(config.d_model, config.ltm_size, config.ltm_top_k, MemoryReplayPolicy[config.replay_policy.upper()])
        self.consciousness_controller = ConsciousnessController(config.d_model, config.consciousness_max_attention_steps)
        self.basal_ganglia = BasalGangliaMechanism(
            d_model=config.d_model,
            action_dim=config.ctm_n_synch_action,
            dopamine_dim=config.ctm_bg_dopamine_dim,
            context_dim=self.d_input
        )
        self.synaptic_empathy = SynapticEmpathy(config.d_model, config.ctm_memory_length, config.n_heads, config.dropout)
        self.mirror_neuron = MirrorNeuronLayer(config.d_model, config.n_heads, config.dropout, config.num_emotion_dim, config.goal_dim)
        self.temporal_spatial_tracker = TemporalSpatialTracker(config)
        self.working_memory = WorkingMemoryBuffer(config.d_model)
        self.glial_support = GlialSupport(config.d_model)
        # --- Input/Output Layers ---
        self.input_encoder = nn.Linear(config.ctm_input_dim, self.d_input)
        self.output_projector = nn.Linear(self.synch_representation_size_out, config.ctm_out_dims)
        
        # --- Initial States ---
        self.start_activated_zL = nn.Parameter(torch.zeros(self.d_model))
        self.start_trace_zL = nn.Parameter(torch.zeros(self.d_model, self.memory_length))
        self.start_zH = nn.Parameter(torch.zeros(self.d_model))
        nn.init.uniform_(self.start_activated_zL, -math.sqrt(1/self.d_model), math.sqrt(1/self.d_model))
        nn.init.uniform_(self.start_trace_zL, -math.sqrt(1/(self.d_model+self.memory_length)), math.sqrt(1/(self.d_model+self.memory_length)))
        nn.init.uniform_(self.start_zH, -math.sqrt(1/self.d_model), math.sqrt(1/self.d_model))
        
        # --- Synchronisation Setup (reusing logic from OriginalCTMCore) ---
        self.neuron_select_type = config.ctm_neuron_select_type
        self.verify_args() # verify neuron selection compatibility
        self.n_synch_out = config.ctm_n_synch_out
        self.n_synch_action = config.ctm_n_synch_action
        self.synch_representation_size_action = self.calculate_synch_representation_size(self.n_synch_action)
        self.synch_representation_size_out = self.calculate_synch_representation_size(self.n_synch_out)
        
        self.set_synchronisation_parameters('action', self.n_synch_action, config.ctm_n_random_pairing_self)
        self.set_synchronisation_parameters('out', self.n_synch_out, config.ctm_n_random_pairing_self)
        self.output_projector = nn.Linear(self.synch_representation_size_out, config.ctm_out_dims)
        self.fusion_proj = nn.Linear(2 * self.d_model, self.d_model)
        
        # Add synchronization parameters for H-module
        self.n_synch_h = config.ctm_n_synch_action  # Reuse action size
        self.synch_representation_size_h = self.calculate_synch_representation_size(self.n_synch_h)
        self.set_synchronisation_parameters('h', self.n_synch_h, config.ctm_n_random_pairing_self)
        
        # --- Quantization Stubs for QAT ---
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()
        
        # --- Bitwidth Adapter for adaptive quantization ---
        if config.ctm_adaptive_quantization:
            self.bitwidth_adapter = BitwidthAdapter(
                self.d_model,
                min_bits=config.ctm_quant_min_bits,
                max_bits=config.ctm_quant_max_bits
            )
        else:
            self.bitwidth_adapter = None
        
        if config.ctm_quant_policy_search:
            # Assuming num_components is related to d_model or another config param
            # For now, let's use a fixed number, e.g., 16 components
            num_quant_components = 16
            self.quant_policy_net = QuantizationPolicyNetwork(
                self.d_model, num_components=num_quant_components
            )
        else:
            self.quant_policy_net = None
        
        if config.ctm_selective_quantization:
            self.selective_quantizer = SelectiveQuantizer(
                min_bits=config.ctm_quant_min_bits,
                max_bits=config.ctm_quant_max_bits,
            )
        else:
            self.selective_quantizer = None

    def forward_with_full_tracking(self,
                                   x: torch.Tensor,
                                   thought_guidance: bool = True,
                                   confidence_level: str = 'medium',
                                   voice1_id: Optional[torch.Tensor] = None,
                                   voice2_id: Optional[torch.Tensor] = None,
                                   blend_degree: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
       """
        The main forward pass implementing the hierarchical reasoning process.
        This method will replace the original CTM's iterative loop.
       """
       self.fusion_proj = nn.Linear(2 * self.d_model, self.d_model)
       b, s, _ = x.shape
       device = x.device
       self.consciousness_controller.wake_up(0)
   
       quantize = (self.config.quant_enabled_training and self.training) or \
                  (self.config.quant_enabled_inference and not self.training)

       if quantize and self.config.ctm_use_qat:
           x = self.quant(x)
    
       # 1. Project input
       x_context = self.input_encoder(x)
       
       # 2. Initialize states
       activated_zL = self.start_activated_zL.unsqueeze(0).expand(b, -1)
       zL_trace = self.start_trace_zL.unsqueeze(0).expand(b, -1, -1)
       zH = self.start_zH.unsqueeze(0).expand(b, -1)
       
       # Optional voice blending
       if voice1_id is not None and voice2_id is not None and blend_degree is not None:
           blended_voice = self.voice_blender(voice1_id, voice2_id, blend_degree)
           zH = zH + blended_voice  # Add blended voice to high-level state
       
       # 3. Initialize sync recurrent values
       decay_alpha_action, decay_beta_action = None, None
       decay_alpha_out, decay_beta_out = None, None
       decay_alpha_h, decay_beta_h = None, None
       r_action = torch.exp(-self.decay_params_action).unsqueeze(0).expand(b, -1)
       r_out = torch.exp(-self.decay_params_out).unsqueeze(0).expand(b, -1)
       r_h = torch.exp(-self.decay_params_h).unsqueeze(0).expand(b, -1)
    
       # Store history of high-level states for final representation
       zH_history = []
       programs = []
       total_entropy_loss = torch.tensor(0.0, device=device)
    
       # 4. Hierarchical recurrent loop
       for n in range(self.config.hrm_high_level_cycles):
           # The L-module's own synchronisation state is reset/recalculated each high-level cycle
           decay_alpha_action, decay_beta_action = None, None
    
           prev_zL = activated_zL.clone()
           for t in range(self.config.hrm_low_level_timesteps):
               # Compute L-module's action synchronisation for its attention query
               sync_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(
                   activated_zL, decay_alpha_action, decay_beta_action, r_action, 'action'
               )
               
               if self.basal_ganglia:
                   action_candidates = [sync_action, sync_action * 0.5, sync_action * 1.5]
                   sync_action = self.basal_ganglia.select_action(action_candidates, activated_zL, x_context.mean(dim=1))
               
               # Run one step of the L-module
               activated_zL, zL_trace = self.l_module(
                   activated_zL, zL_trace, zH, x_context, sync_action, confidence_level=confidence_level
               )
               
               # Apply Neuromodulators to activated_zL
               if self.config.enable_neuromodulators:
                   mod_outputs = [mod(activated_zL) for mod in self.neuromodulators.values()]
                   concatenated_mods = torch.cat(mod_outputs, dim=-1)
                   fused_mod = self.mod_fusion(concatenated_mods)
                   activated_zL = activated_zL * fused_mod
               
               activated_zL = self.working_memory.update(activated_zL)
   
               # Early stopping if change is small
               
               delta = torch.norm(activated_zL - prev_zL)
               if delta < 1e-3:
                   break
               prev_zL = activated_zL.clone()
           
           # Calculate surprise
           surprise = compute_normalized_entropy(activated_zL.unsqueeze(1)).mean()
           
           # Store zH in LTM if surprising
           if surprise > self.config.ltm_surprise_threshold:
               self.ltm.add_to_memory(zH.squeeze(0), surprise)
    
           # Retrieve from LTM
           retrieved_memory = self.ltm.retrieve_from_memory(zH.squeeze(0))
    
           # End of low-level cycle, update high-level state using the final L-state
           # Fuse retrieved_memory with zL before passing to h_module
           # Fuse retrieved_memory with zL before passing to h_module with bidirectional attention
           fused_input = torch.cat([activated_zL.unsqueeze(1), retrieved_memory], dim=1)
           attn_out, _ = nn.MultiheadAttention(self.d_model, num_heads=8, batch_first=True)(fused_input, fused_input, fused_input)
           fused_input = self.fusion_proj(attn_out.mean(dim=1))
           
           zH, encoded_patches, patch_indices, entropy_aux_loss = self.h_module(zH, fused_input, retrieved_memory, thought_guidance=thought_guidance, confidence_level=confidence_level)
           
           # Apply Neuromodulators to zH
           if self.config.enable_neuromodulators:
               mod_outputs = [mod(zH) for mod in self.neuromodulators.values()]
               concatenated_mods = torch.cat(mod_outputs, dim=-1)
               fused_mod = self.mod_fusion(concatenated_mods)
               zH = zH * fused_mod
           
           modulation = self.consciousness_controller.get_attention_modulation()
           zH = zH * modulation
   
           if quantize and self.config.ctm_adaptive_quantization and self.bitwidth_adapter:
               with torch.no_grad():
                   task_embedding = zH.mean(dim=1)  # Per-sample mean
                   bits = self.bitwidth_adapter(task_embedding)
               
               if self.config.ctm_quant_policy_search and self.quant_policy_net:
                   with torch.no_grad():
                       policy_params = self.quant_policy_net(task_embedding)
                       # For simplicity, we'll use the average params for the whole tensor
                       scale = policy_params[:, :, 1].mean()
                       zero_point = policy_params[:, :, 2].mean().round().int()
                   q_zH, _, _ = quantize_adaptive(zH, bits)
                   zH = dequantize_adaptive(q_zH, scale, zero_point)
               else:
                   q_zH, scale, zero_point = quantize_adaptive(zH, bits)
                   zH = dequantize_adaptive(q_zH, scale, zero_point)
               if self.config.ctm_use_qat:
                   zH = self.dequant(zH)
           
           # Apply Synaptic Empathy
           synaptic_modulation, empathy_reward = self.synaptic_empathy(activated_zL.unsqueeze(-1), zH.unsqueeze(-1), zH)
           zH = zH + synaptic_modulation * 0.1
           
           # Apply Mirror Neuron Layer
           valence, arousal = self.mirror_neuron.get_valence_arousal(zH.unsqueeze(1))
           observed_valence, observed_arousal = self.mirror_neuron.get_valence_arousal(zH.unsqueeze(1))  # Placeholder for observed
           emotion = torch.cat([valence, arousal], dim=-1)
           observed_emotion = torch.cat([observed_valence, observed_arousal], dim=-1)
           modulated_zH, _, current_observed_goal, mirror_reward = self.mirror_neuron(
               zH.unsqueeze(1), zH.unsqueeze(1),
               emotion,
               observed_emotion,
               torch.zeros(b, 1, self.config.goal_dim, device=device)
           )
           zH = modulated_zH.squeeze(1)
           
           # Apply Glial Support for state stabilization
           zH = self.glial_support(zH)
           
           # Apply Working Memory
           zH = self.working_memory.update(zH)
           
           # Apply Temporal-Spatial Tracker
           zH = self.temporal_spatial_tracker(zH.unsqueeze(1)).squeeze(1)
           zH_history.append(zH)
           # The 'programs' list now stores the patch embeddings (if not None)
           if encoded_patches is not None:
               programs.append(encoded_patches)
           total_entropy_loss += entropy_aux_loss
    
           # Replay from LTM
           replayed_memory = self.ltm.replay_memory(batch_size=self.replay_batch_size)
           if replayed_memory is not None:
               # Process replayed memory through H-module to make it a compatible state
               # Here we might need a simplified pass or assumption
               # For now, let's assume replayed memory can be treated as a zH-like state
               zH_history.append(replayed_memory.mean(dim=0, keepdim=True).expand(b, -1))
    
       # 5. Compute final output synchronisation from the history of H-states
       # This treats the H-module's evolution as the final 'trace' for output
       final_zH_trace = torch.stack(zH_history, dim=-1) # (B, D, N)
       decay_alpha_out, decay_beta_out = None, None
       for i in range(self.config.hrm_high_level_cycles):
           synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
               final_zH_trace[:,:,i], decay_alpha_out, decay_beta_out, r_out, 'out'
           )
    
       # 6. Project output from the final H-synchronisation
       predictions = self.output_projector(synchronisation_out)
       certainties = self.compute_certainty(predictions)
       
       # Confidence Thresholding
       abstain_mask = torch.zeros(b, dtype=torch.bool, device=device)
       if self.config.confidence_threshold > 0:
           confidence_scores = certainties[:, 1]  # Shape: (B,)
           abstain_mask = confidence_scores < self.config.confidence_threshold
    
       self.consciousness_controller.sleep_down(0)
       return {
           'predictions': predictions.unsqueeze(-1),
           'certainties': certainties.unsqueeze(-1),
           'abstained': abstain_mask.unsqueeze(-1),
           'final_sync_out': synchronisation_out,
           'activated_states': zH_history,
           'programs': programs,
           'entropy_aux_loss': total_entropy_loss
       }
    

class GlialSupport(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.adaptive_norm = nn.LayerNorm(d_model)
                self.support_mlp = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Linear(d_model // 2, d_model)
                )
        
            def forward(self, x):
                norm_x = self.adaptive_norm(x)
                support = self.support_mlp(norm_x)
                return x + support * 0.1


class ForesightSimulator(nn.Module):
    def __init__(self, d_model, num_steps=3):
        super().__init__()
        self.gru = GRU(d_model, d_model, num_layers=1, batch_first=True)
        self.num_steps = num_steps
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, zH):
        batch_size = zH.size(0)
        simulated = zH.unsqueeze(1)
        for _ in range(self.num_steps):
            out, _ = self.gru(simulated)
            simulated = torch.cat([simulated, self.proj(out[:, -1:])], dim=1)
        return simulated[:, 1:].mean(dim=1)  # Average future states

class HyperNetwork(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        param_size = out_features * in_features + out_features
        self.fc = nn.Linear(in_features, param_size)

    def forward(self, x):
        params = self.fc(x)
        weight = params[:, :self.out_features * self.in_features].view(-1, self.out_features, self.in_features)
        bias = params[:, self.out_features * self.in_features:].view(-1, self.out_features)
        return weight, bias


class DynamicEntropyRotationalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.theta_base = 10000.0
        self.L_train = 512
        self.L_target = 4096

    def forward(self, x, entropy_scores, use_rescaled=True):
        B, S, D = x.shape
        if D % 2 != 0:
            raise ValueError("d_model must be even for RoPE")
        position = torch.arange(0, S, dtype=torch.float, device=x.device).unsqueeze(0).repeat(B, 1) # (B, S)
        div_term = torch.exp(torch.arange(0, D//2, dtype=torch.float, device=x.device) * (-math.log(self.theta_base) / D)) # (D/2)
        
        if use_rescaled:
            # LongRoPE2-inspired rescaling integrated with entropy
            extension_ratio = self.L_target / self.L_train
            d_tcd = int(2 * math.ceil(D / 2 * math.log(self.theta_base) * self.L_train / (2 * math.pi)))
            
            average_entropy = entropy_scores.mean()
            d_rcd = d_tcd - int(average_entropy.item() * 10)  # Adjust based on entropy
            d_rcd = max(0, min(d_tcd, d_rcd))
            
            i = torch.arange(0, D//2, device=x.device, dtype=torch.float)
            lambda_i = torch.ones_like(i)
            mask_higher = i >= d_rcd
            lambda_i[mask_higher] = extension_ratio * (1 + (i[mask_higher] - d_rcd) / (D//2 - d_rcd))
            mask_lower = ~mask_higher
            lambda_i[mask_lower] = extension_ratio ** (2 * i[mask_lower] / D)
            
            div_term = div_term / lambda_i
        
        theta = position.unsqueeze(-1) * div_term.unsqueeze(0).unsqueeze(1) # (B, S, D/2)
        entropy_scores = entropy_scores.unsqueeze(-1) # (B, S, 1)
        theta = theta * entropy_scores
        cos = torch.cos(theta) # (B, S, D/2)
        sin = torch.sin(theta)
        x0 = x[:, :, 0::2]
        x1 = x[:, :, 1::2]
        x_rot0 = x0 * cos - x1 * sin
        x_rot1 = x0 * sin + x1 * cos
        x_rot = torch.zeros_like(x)
        x_rot[:, :, 0::2] = x_rot0
        x_rot[:, :, 1::2] = x_rot1
        return self.dropout(x_rot)

class ProgramSynthesizer(nn.Module):
    """
    A transformer-based program synthesizer that generates a program as a sequence of bytes,
    and then segments it into entropy-based patches.
    
    This module takes a high-level state representation, generates a raw byte sequence
    autoregressively, and then uses a DynamicEntropyPatcher to structure the sequence.
    """
    def __init__(self, 
                 d_model: int, 
                 n_heads: int = 4, 
                 n_layers: int = 3,
                 d_ff: int = 1024,
                 dropout: float = 0.1,
                 max_gen_len: int = 512,
                 patcher_config: dict = None):
        super().__init__()
        self.d_model = d_model
        self.max_gen_len = max_gen_len
        self.byte_vocab_size = 256

        # --- Transformer Decoder for Byte Generation ---
        # We need a decoder, not an encoder for generation.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # --- I/O Layers ---
        self.byte_embedding = nn.Embedding(self.byte_vocab_size, d_model)
        self.output_projector = nn.Linear(d_model, self.byte_vocab_size)
        
        # --- Positional Encoding ---
        self.pos_encoder = DynamicEntropyRotationalEmbedding(d_model, dropout)
        
        # --- Entropy-based Patcher ---
        # Use provided config or a default for the patcher
        if patcher_config is None:
            patcher_config = {
                'embedding_dim': d_model,
                'patch_cnn_channels': 64,
                'patching_mode': "global",
                'global_threshold': 0.5,
                'relative_threshold': 0.1,
                'min_patch_size': 4,
                'max_patch_size': 128,
                'entropy_byte_vocab_size': 256,
                'entropy_embedding_dim': 64,
                'entropy_hidden_dim': 128,
                'entropy_num_layers': 1,
                'entropy_dropout': 0.1
            }
        self.patcher = DynamicEntropyPatcher(**patcher_config)

        # --- Neuron Pulsing Integration ---
        self.ctm_config = EnhancedCTMConfig(
            ctm_iterations=5,
            ctm_d_model=self.d_model,
            ctm_input_dim=self.d_model,
            ctm_heads=4,
            ctm_n_synch_out=256,
            ctm_n_synch_action=64,
            ctm_synapse_depth=1,
            ctm_memory_length=10,
            ctm_deep_nlms=True,
            ctm_memory_hidden_dims=512,
            ctm_do_layernorm_nlm=False,
            ctm_out_dims=256,
            ctm_prediction_reshaper=[-1, 256],
            ctm_dropout=0.1,
            ctm_neuron_select_type='random-pairing'
        )
        self.ctm_core = OriginalCTMCore(self.ctm_config)
        self.byte_predictor = nn.Linear(256, 256)  # from ctm_out_dims to byte_vocab

    def forward(self, state: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[List[Tuple[int, int]]], torch.Tensor]:
        """
        Synthesizes a program as a sequence of byte patches from a given state using neuron pulsing.
        This version runs the internal CTM for its full duration to generate a thought vector,
        which is then projected to the full byte sequence.
        
        Args:
            state: A tensor representing the high-level state.
                   Shape: (batch_size, d_model)
            attn_mask: Optional attention mask for packed sequences.
                   
        Returns:
            A tuple containing:
            - encoded_patches: Tensor of patch embeddings. (B, max_patches, patch_embed_dim)
            - patch_indices: List of (start, end) indices for each patch.
            - entropy_aux_loss: Auxiliary loss from the patcher's entropy model.
        """
        device = state.device
        batch_size = state.size(0)
        
        # The context for the synthesizer's CTM is the input high-level state.
        # Shape: (B, 1, D) to represent a sequence of length 1.
        kv = state.unsqueeze(1)

        # Run the internal CTM for its full "thought" process.
        # The CTM's internal loop handles the "pulsing".
        # We use the final synchronisation vector as the program representation.
        ctm_predictions, ctm_certainties, ctm_sync_out = self.ctm_core(kv)

        # Project the final thought vector (sync_out) to the entire byte sequence.
        # This requires a different predictor that can handle the full sequence length.
        # Let's adjust the byte_predictor to output the whole sequence.
        # For now, we project to max_gen_len * vocab_size and reshape.
        if not hasattr(self, 'sequence_predictor'):
            self.sequence_predictor = nn.Linear(ctm_sync_out.shape[-1], self.max_gen_len * self.byte_vocab_size).to(device)

        logits_flat = self.sequence_predictor(ctm_sync_out)
        logits = logits_flat.view(batch_size, self.max_gen_len, self.byte_vocab_size)

        # Get the generated byte sequence by taking argmax
        generated_bytes = torch.argmax(logits, dim=-1) # (B, max_gen_len)
        
        # Patch the generated bytes
        encoded_patches, patch_indices, entropy_aux_loss = self.patcher(generated_bytes.byte())
        
        return encoded_patches, patch_indices, entropy_aux_loss

    
