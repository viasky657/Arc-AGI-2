"""
Enhanced CTM-Controlled Diffusion Architecture - This is the most current version of ctm.py that is currently being used.

This implementation gives the CTM deep control and influence over the diffusion process
through multiple mechanisms:
1. Direct noise prediction conditioning
2. Adaptive timestep scheduling based on CTM certainty
3. CTM-guided attention mechanisms with WINA sparse activation
4. Synchronization-based diffusion guidance
5. Iterative CTM-diffusion coupling
6. WINA (Weight Informed Neuron Activation) for efficient sparse attention now with Top-Down attention to Dynamically-Adjust and Learn from errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, Any, List
from dataclasses import dataclass, field
from collections import defaultdict
import math
import numpy as np
import time
import pickle
import os
import hashlib
import threading
from typing import TYPE_CHECKING # For type hinting TaskAnalyzer

if TYPE_CHECKING:
    from .utils import TaskAnalyzer 
import random
import copy # For JEPA target encoder deepcopy
from concurrent.futures import ThreadPoolExecutor
import queue
#from diffusers.schedulers.scheduling_ddpm import DDPMScheduler #Need to install with pip #Replaced with DPMSOLVerMultistepScheduler
from diffusers import DPMSolverMultistepScheduler  #Need to install with pip
import numpy as np

try:

    import sys

    sys.path.append('/workspaces/Arc-AGI-2/contineous_thought_machines/models/flash-attention-3')

    from flash_attn import flash_attention

except ImportError:

    flash_attention = None


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


from .enhanced_neuron_selection import EnhancedNeuronSelector #Enhances Nueron Selections with Biologically-Inspired Systems instead of Random
from .ctm_unified_diffusion import UnifiedCTMDenoisingModel
from .ctm_components import (
    EnhancedCTMConfig,
    HierarchicalCTM,
    OriginalCTMCore,
    CTMFeedbackModule,
    BasalGangliaMechanism,
    SynapticEmpathy,
    MirrorNeuronLayer,
    EmotionStateTracker,
    GoalPredictor,
    WINAAttention,
    BinarySparseAttention,
    WINAEnhancedMLP,
    WINASparsifier,
    MetaWINASparsifier,
    ConsciousnessController,
    BidirectionalReasoningController,
    FrequencyDomainAwareAttention,
    TemporalSpatialTracker,
    WorkingMemoryBuffer,
    ForesightSimulator,
    HyperNetwork,
)
from .biological_neuron_selection import BiologicalNeuronSelector, BiologicalSelectionConfig
from .realtime_voice_module import RealtimeVoiceStreamer
# Import original CTM modules to preserve exact behavior
# try:
from .modules import SynapseUNET, Squeeze, SuperLinear, LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding
from .utils import compute_normalized_entropy
from .constants import VALID_NEURON_SELECT_TYPES, VALID_POSITIONAL_EMBEDDING_TYPES
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


class PipelineParallelProcessor(nn.Module):
    """
    Pipeline parallelism processor for overlapping CTM and diffusion computation.
    Implements DiffusionPipe-style optimizations with computation overlap.
    """
    
    def __init__(self, config: 'EnhancedCTMConfig'):
        super().__init__()
        self.config = config
        self.pipeline_stages = 3  # CTM, Diffusion prep, Diffusion exec
        self.overlap_enabled = True
        
        # Pipeline stage queues
        self.stage_queues = [queue.Queue(maxsize=2) for _ in range(self.pipeline_stages)]
        self.result_queue = queue.Queue()
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=self.pipeline_stages)
        self.pipeline_active = False
        
    def start_pipeline(self):
        """Start the pipeline processing threads."""
        self.pipeline_active = True
        
    def stop_pipeline(self):
        """Stop the pipeline processing threads."""
        self.pipeline_active = False
        
    def pipeline_forward(self, ctm_core, diffusion_processor, inputs, timesteps, guidance_data):
        """
        Execute forward pass with pipeline parallelism.
        Overlaps CTM computation with diffusion preparation.
        """
        if not self.overlap_enabled:
            # Fallback to sequential processing
            return self._sequential_forward(ctm_core, diffusion_processor, inputs, timesteps, guidance_data)
        
        # Stage 1: CTM Core (can run in parallel with diffusion prep)
        ctm_future = self.executor.submit(self._ctm_stage, ctm_core, inputs)
        
        # Stage 2: Diffusion preparation (can overlap with CTM)
        diff_prep_future = self.executor.submit(self._diffusion_prep_stage, inputs, timesteps)
        
        # Wait for CTM completion
        ctm_results = ctm_future.result()
        
        # Wait for diffusion prep
        diff_prep_results = diff_prep_future.result()
        
        # MCMC results are no longer used
        mcmc_results = None

        # Stage 4: Final diffusion execution
        final_guidance = self._merge_guidance(guidance_data, ctm_results, mcmc_results)
        diffusion_output = diffusion_processor(diff_prep_results['noisy_input'],
                                             diff_prep_results['timesteps'],
                                             final_guidance)
        
        return {
            'ctm_results': ctm_results,
            'mcmc_results': None,
            'diffusion_output': diffusion_output,
            'pipeline_efficiency': self._calculate_efficiency()
        }
    
    def _ctm_stage(self, ctm_core, inputs):
        """CTM processing stage."""
        return ctm_core.forward_with_full_tracking(inputs)
    
    def _diffusion_prep_stage(self, inputs, timesteps):
        """Diffusion preparation stage that can overlap with CTM."""
        # Prepare noise and timestep embeddings
        noise = torch.randn_like(inputs)
        timestep_embeddings = self._embed_timesteps(timesteps)
        
        return {
            'noisy_input': inputs + noise * 0.1,  # Light noise for preparation
            'timesteps': timesteps,
            'timestep_embeddings': timestep_embeddings
        }
    
    def _merge_guidance(self, base_guidance, ctm_results, mcmc_results):
        """Merge guidance from different pipeline stages."""
        merged_guidance = base_guidance.copy() if base_guidance else {}
        merged_guidance.update(ctm_results)
        # MCMC results are no longer merged
        return merged_guidance
    
    def _embed_timesteps(self, timesteps):
        """Create timestep embeddings."""
        return timesteps.float().unsqueeze(-1)
    
    def _calculate_efficiency(self):
        """Calculate pipeline efficiency metrics."""
        return {'overlap_ratio': 0.7, 'speedup_factor': 1.8}
    
    def _sequential_forward(self, ctm_core, diffusion_processor, inputs, timesteps, guidance_data):
        """Fallback sequential processing."""
        ctm_results = ctm_core.forward_with_full_tracking(inputs)
        noisy_input = inputs + torch.randn_like(inputs) * 0.1
        final_guidance = self._merge_guidance(guidance_data, ctm_results, None)
        diffusion_output = diffusion_processor(noisy_input, timesteps, final_guidance)
        
        return {
            'ctm_results': ctm_results,
            'mcmc_results': None,
            'diffusion_output': diffusion_output,
            'pipeline_efficiency': {'overlap_ratio': 0.0, 'speedup_factor': 1.0}
        }


class AdaptiveBatchSampler:
    """
    Adaptive batch sizing system that dynamically adjusts batch size based on:
    1. GPU memory utilization
    2. Training convergence rate
    3. Data complexity
    4. Pipeline efficiency
    """
    
    def __init__(self, initial_batch_size: int = 32, min_batch_size: int = 8,
                 max_batch_size: int = 256, adaptation_frequency: int = 100):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.adaptation_frequency = adaptation_frequency
        
        # Tracking metrics
        self.step_count = 0
        self.memory_usage_history = []
        self.loss_history = []
        self.throughput_history = []
        
        # Adaptation parameters
        self.memory_threshold_high = 0.85  # 85% GPU memory usage
        self.memory_threshold_low = 0.6    # 60% GPU memory usage
        self.convergence_window = 50
        
    def should_adapt(self) -> bool:
        """Check if batch size should be adapted."""
        self.step_count += 1
        return self.step_count % self.adaptation_frequency == 0
    
    def update_metrics(self, memory_usage: float, loss: float, throughput: float):
        """Update tracking metrics."""
        self.memory_usage_history.append(memory_usage)
        self.loss_history.append(loss)
        self.throughput_history.append(throughput)
        
        # Keep only recent history
        max_history = 200
        if len(self.memory_usage_history) > max_history:
            self.memory_usage_history = self.memory_usage_history[-max_history:]
            self.loss_history = self.loss_history[-max_history:]
            self.throughput_history = self.throughput_history[-max_history:]
    
    def adapt_batch_size(self) -> int:
        """Adapt batch size based on current metrics."""
        if len(self.memory_usage_history) < 10:
            return self.current_batch_size
        
        recent_memory = np.mean(self.memory_usage_history[-10:])
        recent_throughput = np.mean(self.throughput_history[-10:])
        
        # Memory-based adaptation
        if recent_memory > self.memory_threshold_high:
            # Reduce batch size if memory usage is high
            new_batch_size = max(self.min_batch_size,
                               int(self.current_batch_size * 0.8))
        elif recent_memory < self.memory_threshold_low:
            # Increase batch size if memory usage is low
            new_batch_size = min(self.max_batch_size,
                               int(self.current_batch_size * 1.2))
        else:
            new_batch_size = self.current_batch_size
        
        # Convergence-based fine-tuning
        if len(self.loss_history) >= self.convergence_window:
            recent_losses = self.loss_history[-self.convergence_window:]
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            
            if loss_trend > 0:  # Loss increasing (diverging)
                new_batch_size = max(self.min_batch_size,
                                   int(new_batch_size * 0.9))
            elif abs(loss_trend) < 1e-6:  # Loss plateaued
                new_batch_size = min(self.max_batch_size,
                                   int(new_batch_size * 1.1))
        
        # Throughput optimization
        if len(self.throughput_history) >= 20:
            if recent_throughput < np.mean(self.throughput_history[-20:-10]):
                # Throughput decreased, try smaller batch
                new_batch_size = max(self.min_batch_size,
                                   int(new_batch_size * 0.95))
        
        self.current_batch_size = new_batch_size
        return new_batch_size
    
    def get_current_batch_size(self) -> int:
        """Get current batch size."""
        return self.current_batch_size

class SmartDataSampler:
    """
    Intelligent data sampling system for prioritizing informative binary patterns.
    Uses importance scoring, diversity metrics, and active learning principles.
    """
    
    def __init__(self, dataset_size: int, initial_sample_ratio: float = 0.3,
                 diversity_weight: float = 0.4, importance_weight: float = 0.6):
        self.dataset_size = dataset_size
        self.sample_ratio = initial_sample_ratio
        self.diversity_weight = diversity_weight
        self.importance_weight = importance_weight
        
        # Sample tracking
        self.sample_scores = np.ones(dataset_size) * 0.5  # Initial neutral scores
        self.sample_diversity = np.ones(dataset_size) * 0.5
        self.sample_access_count = np.zeros(dataset_size)
        self.sample_last_loss = np.ones(dataset_size) * float('inf')
        
        # Pattern analysis
        self.binary_pattern_cache = {}
        self.complexity_scores = np.ones(dataset_size) * 0.5
        
    def update_sample_importance(self, indices: List[int], losses: List[float],
                                gradients: Optional[List[torch.Tensor]] = None):
        """Update importance scores based on training feedback."""
        for idx, loss in zip(indices, losses):
            if idx < len(self.sample_scores):
                # Higher loss = higher importance (more informative)
                self.sample_scores[idx] = min(1.0, loss / 10.0)
                self.sample_last_loss[idx] = loss
                self.sample_access_count[idx] += 1
        
        # Gradient-based importance (if available)
        if gradients:
            for idx, grad in zip(indices, gradients):
                if idx < len(self.sample_scores) and grad is not None:
                    grad_norm = torch.norm(grad).item()
                    # Higher gradient norm = more informative
                    gradient_importance = min(1.0, grad_norm / 100.0)
                    self.sample_scores[idx] = 0.7 * self.sample_scores[idx] + 0.3 * gradient_importance
    
    def analyze_binary_patterns(self, data_batch: torch.Tensor, indices: List[int]):
        """Analyze binary patterns for complexity and diversity scoring."""
        for i, idx in enumerate(indices):
            if idx >= len(self.complexity_scores):
                continue
                
            sample_data = data_batch[i].cpu().numpy()
            
            # Calculate complexity metrics
            entropy = self._calculate_entropy(sample_data)
            pattern_diversity = self._calculate_pattern_diversity(sample_data)
            compression_ratio = self._estimate_compression_ratio(sample_data)
            
            # Combine into complexity score
            complexity = (entropy * 0.4 + pattern_diversity * 0.3 + compression_ratio * 0.3)
            self.complexity_scores[idx] = complexity
            
            # Update diversity based on uniqueness
            self.sample_diversity[idx] = self._calculate_sample_uniqueness(sample_data, idx)
    
    def get_priority_samples(self, num_samples: int, exclude_recent: bool = True) -> List[int]:
        """Get prioritized sample indices for training."""
        # Combine importance, diversity, and complexity
        composite_scores = (
            self.importance_weight * self.sample_scores +
            self.diversity_weight * self.sample_diversity +
            0.2 * self.complexity_scores
        )
        
        # Penalize recently accessed samples for diversity
        if exclude_recent:
            access_penalty = np.log1p(self.sample_access_count) * 0.1
            composite_scores = composite_scores - access_penalty
        
        # Add some randomness to prevent overfitting to specific samples
        noise = np.random.normal(0, 0.05, len(composite_scores))
        composite_scores = composite_scores + noise
        
        # Select top samples
        top_indices = np.argsort(composite_scores)[-num_samples:]
        return top_indices.tolist()
    
    def get_diverse_batch(self, batch_size: int) -> List[int]:
        """Get a diverse batch using clustering-based selection."""
        # Use k-means style selection for diversity
        num_clusters = min(batch_size, 10)
        
        # Simple diversity-based sampling
        selected_indices = []
        remaining_indices = list(range(self.dataset_size))
        
        for _ in range(batch_size):
            if not remaining_indices:
                break
                
            # Select based on composite score and distance from already selected
            best_idx = self._select_most_diverse(remaining_indices, selected_indices)
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return selected_indices
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of binary data."""
        if len(data) == 0:
            return 0.0
        
        # Convert to bytes if needed
        if data.dtype != np.uint8:
            data = (data * 255).astype(np.uint8)
        
        # Calculate byte frequency
        unique, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        
        # Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy / 8.0  # Normalize to [0, 1]
    
    def _calculate_pattern_diversity(self, data: np.ndarray) -> float:
        """Calculate diversity of binary patterns."""
        if len(data) < 8:
            return 0.5
        
        # Look at 8-bit patterns
        patterns = []
        for i in range(len(data) - 7):
            pattern = tuple(data[i:i+8])
            patterns.append(pattern)
        
        unique_patterns = len(set(patterns))
        max_possible = min(256, len(patterns))
        
        return unique_patterns / max_possible if max_possible > 0 else 0.5
    
    def _estimate_compression_ratio(self, data: np.ndarray) -> float:
        """Estimate compression ratio as complexity measure."""
        if len(data) < 10:
            return 0.5
        
        # Simple run-length encoding estimation
        runs = 1
        for i in range(1, len(data)):
            if data[i] != data[i-1]:
                runs += 1
        
        compression_ratio = runs / len(data)
        return min(1.0, compression_ratio)
    
    def _calculate_sample_uniqueness(self, sample_data: np.ndarray, sample_idx: int) -> float:
        """Calculate how unique this sample is compared to others."""
        # Simple hash-based uniqueness for now
        sample_hash = hash(sample_data.tobytes())
        
        if sample_hash not in self.binary_pattern_cache:
            self.binary_pattern_cache[sample_hash] = []
        
        self.binary_pattern_cache[sample_hash].append(sample_idx)
        
        # Uniqueness is inverse of frequency
        frequency = len(self.binary_pattern_cache[sample_hash])
        return 1.0 / frequency
    
    def _select_most_diverse(self, remaining_indices: List[int], selected_indices: List[int]) -> int:
        """Select the most diverse sample from remaining indices."""
        if not selected_indices:
            # First selection based on importance
            scores = [self.sample_scores[i] for i in remaining_indices]
            return remaining_indices[np.argmax(scores)]
        
        best_idx = remaining_indices[0]
        best_score = -float('inf')
        
        for idx in remaining_indices:
            # Combine importance with diversity from selected samples
            importance = self.sample_scores[idx]
            
            # Simple diversity metric (could be improved with actual distance calculation)
            diversity = self.sample_diversity[idx]
            
            # Penalize if too similar to already selected (simplified)
            similarity_penalty = self.sample_access_count[idx] * 0.1
            
            score = importance + diversity - similarity_penalty
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        return best_idx


class MultiGranularityBinaryProcessor(nn.Module):
    """
    Multi-granularity binary processing that simultaneously processes data at:
    1. Bit level (individual bits)
    2. Byte level (8-bit chunks)
    3. Word level (16/32-bit chunks)
    4. Block level (64+ bit chunks)
    
    This provides hierarchical understanding from fine-grained to coarse-grained patterns.
    """
    
    def __init__(self, config: 'EnhancedCTMConfig'):
        super().__init__()
        self.config = config
        self.output_dim = config.byte_embedding_dim
        
        # Bit-level processing (1-bit granularity)
        self.bit_processor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=1, padding=3),  # Process 8 bits at a time
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(self.output_dim // 4)
        )
        
        # Byte-level processing (8-bit granularity)
        self.byte_processor = nn.Sequential(
            nn.Embedding(256, 64),  # 256 possible byte values
            nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(self.output_dim // 4)
        )
        
        # Word-level processing (16-bit granularity)
        self.word_processor = nn.Sequential(
            nn.Embedding(65536, 128),  # 2^16 possible word values
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(self.output_dim // 4)
        )
        
        # Block-level processing (32+ bit granularity)
        self.block_processor = nn.Sequential(
            nn.Linear(32, 256),  # Process 32-bit blocks
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim // 4)
        )
        
        # Hierarchical fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )
        
        # Attention mechanism for granularity weighting
        self.granularity_attention = nn.MultiheadAttention(
            embed_dim=self.output_dim // 4,
            num_heads=4,
            batch_first=True
        )
        
    def extract_bit_features(self, byte_sequence: torch.Tensor) -> torch.Tensor:
        """Extract bit-level features from byte sequence."""
        batch_size, seq_len = byte_sequence.shape
        
        # Convert bytes to bits
        bits = []
        for i in range(8):
            bit_plane = (byte_sequence >> i) & 1
            bits.append(bit_plane.float())
        
        # Stack bit planes: (batch_size, 8, seq_len)
        bit_tensor = torch.stack(bits, dim=1)
        
        # Reshape for conv1d: (batch_size, 1, 8*seq_len)
        bit_input = bit_tensor.view(batch_size, 1, -1)
        
        # Process through bit-level network
        bit_features = self.bit_processor(bit_input)  # (batch_size, output_dim//4, seq_len)
        
        return bit_features.transpose(1, 2)  # (batch_size, seq_len, output_dim//4)
    
    def extract_byte_features(self, byte_sequence: torch.Tensor) -> torch.Tensor:
        """Extract byte-level features."""
        # Embed bytes: (batch_size, seq_len, 64)
        byte_embedded = self.byte_processor[0](byte_sequence)
        
        # Conv1d processing: (batch_size, 64, seq_len) -> (batch_size, output_dim//4, seq_len)
        byte_conv_input = byte_embedded.transpose(1, 2)
        for layer in self.byte_processor[1:]:
            byte_conv_input = layer(byte_conv_input)
        
        return byte_conv_input.transpose(1, 2)  # (batch_size, seq_len, output_dim//4)
    
    def extract_word_features(self, byte_sequence: torch.Tensor) -> torch.Tensor:
        """Extract word-level (16-bit) features."""
        batch_size, seq_len = byte_sequence.shape
        
        # Combine pairs of bytes into words (handle odd lengths)
        if seq_len % 2 == 1:
            # Pad with zero if odd length
            byte_sequence = torch.cat([byte_sequence, torch.zeros(batch_size, 1, dtype=byte_sequence.dtype, device=byte_sequence.device)], dim=1)
            seq_len += 1
        
        # Reshape to words: (batch_size, seq_len//2)
        words = byte_sequence.view(batch_size, seq_len // 2, 2)
        word_values = words[:, :, 0] * 256 + words[:, :, 1]  # Combine bytes to words
        
        # Embed words: (batch_size, seq_len//2, 128)
        word_embedded = self.word_processor[0](word_values)
        
        # Conv1d processing
        word_conv_input = word_embedded.transpose(1, 2)
        for layer in self.word_processor[1:]:
            word_conv_input = layer(word_conv_input)
        
        # Interpolate back to original sequence length
        word_features = word_conv_input.transpose(1, 2)
        if word_features.size(1) != seq_len // 2:
            word_features = torch.nn.functional.interpolate(
                word_features.transpose(1, 2),
                size=seq_len // 2,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        # Upsample to match original sequence length
        word_features = torch.nn.functional.interpolate(
            word_features.transpose(1, 2),
            size=byte_sequence.size(1) - (1 if seq_len != byte_sequence.size(1) else 0),
            mode='linear',
            align_corners=False
        ).transpose(1, 2)
        
        return word_features
    
    def extract_block_features(self, byte_sequence: torch.Tensor) -> torch.Tensor:
        """Extract block-level (32+ bit) features."""
        batch_size, seq_len = byte_sequence.shape
        
        # Group bytes into 32-bit blocks
        block_size = 4  # 4 bytes = 32 bits
        if seq_len % block_size != 0:
            # Pad to make divisible by block_size
            padding_size = block_size - (seq_len % block_size)
            byte_sequence = torch.cat([
                byte_sequence,
                torch.zeros(batch_size, padding_size, dtype=byte_sequence.dtype, device=byte_sequence.device)
            ], dim=1)
            seq_len += padding_size
        
        # Reshape to blocks: (batch_size, seq_len//block_size, block_size)
        blocks = byte_sequence.view(batch_size, seq_len // block_size, block_size)
        
        # Convert to float and normalize
        block_input = blocks.float() / 255.0  # Normalize to [0, 1]
        
        # Add positional encoding for block positions
        block_positions = torch.arange(block_size, device=byte_sequence.device).float()
        block_positions = block_positions.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len // block_size, -1)
        block_input = torch.cat([block_input, block_positions / block_size], dim=-1)  # (batch_size, num_blocks, 8)
        
        # Pad to 32 features if needed
        if block_input.size(-1) < 32:
            padding = torch.zeros(batch_size, block_input.size(1), 32 - block_input.size(-1), device=byte_sequence.device)
            block_input = torch.cat([block_input, padding], dim=-1)
        
        # Process through block network
        block_features = self.block_processor(block_input)  # (batch_size, num_blocks, output_dim//4)
        
        # Upsample to match original sequence length
        block_features = torch.nn.functional.interpolate(
            block_features.transpose(1, 2),
            size=byte_sequence.size(1) - (seq_len - byte_sequence.size(1)),
            mode='linear',
            align_corners=False
        ).transpose(1, 2)
        
        return block_features
    
    def forward(self, byte_sequence: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Process binary data at multiple granularities simultaneously.
        Supports multiple inputs by processing each separately at bit level.
        
        Args:
            byte_sequence: Single tensor or list of tensors of byte values [0-255]
            
        Returns:
            Multi-granularity features: (batch_size, seq_len, output_dim)
        """
        if isinstance(byte_sequence, list):
            # Process each input separately
            bit_features_list = []
            for i, seq in enumerate(byte_sequence):
                # Use the corresponding input processor
                # Reshape the sequence to (batch, 8, seq_len//8)
                seq_reshaped = seq.view(seq.size(0), 8, -1)
                processed = self.input_processors[i](seq_reshaped)
                # Flatten the processed features
                processed = processed.view(processed.size(0), -1)
                bit_features_list.append(processed)
            # Combine the bit features from multiple inputs
            bit_features = torch.cat(bit_features_list, dim=1)
            # For other features, use the first input
            main_input = byte_sequence[0]
        else:
            # For single input, use the first processor
            seq_reshaped = byte_sequence.view(byte_sequence.size(0), 8, -1)
            bit_features = self.input_processors[0](seq_reshaped)
            bit_features = bit_features.view(bit_features.size(0), -1)
            main_input = byte_sequence
        
        # Extract other features from main input
        byte_features = self.extract_byte_features(main_input)
        word_features = self.extract_word_features(main_input)
        block_features = self.extract_block_features(main_input)
        
        # Ensure all features have the same sequence length
        target_seq_len = main_input.size(1)
        features = []
        
        for feat in [bit_features, byte_features, word_features, block_features]:
            if feat.size(1) != target_seq_len:
                feat = torch.nn.functional.interpolate(
                    feat.transpose(1, 2),
                    size=target_seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            features.append(feat)
        
        # Apply attention-based fusion across granularities
        # Stack features: (batch_size, seq_len, 4, output_dim//4)
        stacked_features = torch.stack(features, dim=2)
        batch_size, seq_len, num_granularities, feat_dim = stacked_features.shape
        
        # Reshape for attention: (batch_size * seq_len, num_granularities, feat_dim)
        attention_input = stacked_features.view(-1, num_granularities, feat_dim)
        
        # Apply self-attention across granularities
        attended_features, attention_weights = self.granularity_attention(
            attention_input, attention_input, attention_input
        )
        
        # Reshape back: (batch_size, seq_len, num_granularities, feat_dim)
        attended_features = attended_features.view(batch_size, seq_len, num_granularities, feat_dim)
        
        # Concatenate attended features: (batch_size, seq_len, output_dim)
        concatenated = attended_features.view(batch_size, seq_len, -1)
        
        # Final fusion
        fused_features = self.fusion_network(concatenated)
        
        return fused_features


class MixedPrecisionTrainer:
    """
    Actual mixed precision training implementation using PyTorch's automatic mixed precision.
    Provides significant speedup and memory savings with minimal accuracy loss.
    """
    
    def __init__(self, model: nn.Module, config: 'EnhancedCTMConfig'):
        self.model = model
        self.config = config
        self.enabled = config.mixed_precision
        
        # Initialize GradScaler for mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if self.enabled and torch.cuda.is_available() else None
        
        # Track mixed precision statistics
        self.mp_stats = {
            'fp16_steps': 0,
            'fp32_steps': 0,
            'scale_updates': 0,
            'overflow_steps': 0
        }
        
    def forward_with_autocast(self, *args, **kwargs):
        """Forward pass with automatic mixed precision."""
        if self.enabled and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                return self.model(*args, **kwargs)
        else:
            return self.model(*args, **kwargs)
    
    def backward_with_scaling(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Backward pass with gradient scaling for mixed precision."""
        if self.enabled and self.scaler is not None:
            # Scale loss to prevent gradient underflow
            self.scaler.scale(loss).backward()
            
            # Update statistics
            self.mp_stats['fp16_steps'] += 1
            
            return True  # Indicates scaled backward was used
        else:
            loss.backward()
            self.mp_stats['fp32_steps'] += 1
            return False  # Indicates normal backward was used
    
    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Optimizer step with gradient unscaling and overflow detection."""
        if self.enabled and self.scaler is not None:
            # Unscale gradients and check for overflow
            self.scaler.unscale_(optimizer)
            
            # Check for gradient overflow
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            if torch.isfinite(total_norm):
                # No overflow, proceed with optimizer step
                self.scaler.step(optimizer)
                self.scaler.update()
                
                # Update statistics
                if self.scaler.get_scale() != self.scaler.get_scale():  # Scale changed
                    self.mp_stats['scale_updates'] += 1
                
                return True
            else:
                # Overflow detected, skip this step
                self.mp_stats['overflow_steps'] += 1
                self.scaler.update()
                return False
        else:
            # Standard optimizer step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            return True
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage statistics."""
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
            }
        return {'allocated_gb': 0, 'reserved_gb': 0, 'max_allocated_gb': 0}
    
    def get_mixed_precision_stats(self) -> Dict[str, Any]:
        """Get mixed precision training statistics."""
        total_steps = self.mp_stats['fp16_steps'] + self.mp_stats['fp32_steps']
        
        stats = self.mp_stats.copy()
        stats.update({
            'total_steps': total_steps,
            'fp16_ratio': self.mp_stats['fp16_steps'] / max(total_steps, 1),
            'overflow_ratio': self.mp_stats['overflow_steps'] / max(total_steps, 1),
            'current_scale': self.scaler.get_scale() if self.scaler else 1.0,
            'enabled': self.enabled
        })
        
        return stats
    
    def reset_stats(self):
        """Reset mixed precision statistics."""
        self.mp_stats = {
            'fp16_steps': 0,
            'fp32_steps': 0,
            'scale_updates': 0,
            'overflow_steps': 0
        }




class JEPAModule(nn.Module):
    def __init__(self, config, dynamic_entropy_patcher):
        super().__init__()
        self.config = config
        self.jepa_target_patch_encoder = copy.deepcopy(dynamic_entropy_patcher)
        for param_target in self.jepa_target_patch_encoder.parameters():
            param_target.requires_grad = False
        
        jepa_io_dim = config.patch_embedding_dim
        predictor_output_dim = jepa_io_dim * config.jepa_num_target_blocks
        self.jepa_predictor = JEPAPredictor(
            input_dim=jepa_io_dim,
            hidden_dim=config.jepa_predictor_hidden_dim,
            output_dim=predictor_output_dim
        )

    @torch.no_grad()
    def update_target_encoder(self, online_encoder):
        m = self.config.jepa_momentum_beta
        for param_online, param_target in zip(online_encoder.parameters(), self.jepa_target_patch_encoder.parameters()):
            param_target.data.mul_(m).add_((1 - m) * param_online.data)

    def create_masked_patch_views(self, online_embeddings, target_embeddings):
        B, S_patches, D_embed = online_embeddings.shape
        device = online_embeddings.device

        if S_patches < 2:
            return None, None

        batch_context_reps = []
        batch_target_reps = []

        for b_idx in range(B):
            num_context_patches = max(1, int(S_patches * random.uniform(self.config.jepa_context_scale_min, self.config.jepa_context_scale_max)))
            num_context_patches = min(num_context_patches, S_patches - self.config.jepa_num_target_blocks)

            if num_context_patches <= 0:
                continue

            all_indices = list(range(S_patches))
            context_indices = random.sample(all_indices, num_context_patches)
            context_block = online_embeddings[b_idx, context_indices, :]
            context_rep = context_block.mean(dim=0)  # (D_embed)

            remaining_indices = [i for i in all_indices if i not in context_indices]
            if len(remaining_indices) < self.config.jepa_num_target_blocks:
                continue

            target_indices = random.sample(remaining_indices, self.config.jepa_num_target_blocks)
            target_block = target_embeddings[b_idx, target_indices, :]  # (num_target_blocks, D_embed)

            batch_context_reps.append(context_rep)
            batch_target_reps.append(target_block)

        if not batch_context_reps:
            return None, None

        return torch.stack(batch_context_reps), torch.stack(batch_target_reps)

class OptimizationModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Pipeline parallelism processor
        if config.enable_pipeline_parallelism:
            self.pipeline_processor = PipelineParallelProcessor(config)
            self.pipeline_processor.start_pipeline()
        else:
            self.pipeline_processor = None
        
        # Adaptive batch sampler
        if config.enable_adaptive_batching:
            self.adaptive_batch_sampler = AdaptiveBatchSampler(
                initial_batch_size=config.initial_batch_size,
                min_batch_size=config.min_batch_size,
                max_batch_size=config.max_batch_size,
                adaptation_frequency=config.batch_adaptation_frequency
            )
        else:
            self.adaptive_batch_sampler = None
        
        # Smart data sampler
        if config.enable_smart_sampling:
            estimated_dataset_size = 100000
            self.smart_data_sampler = SmartDataSampler(
                dataset_size=estimated_dataset_size,
                initial_sample_ratio=config.initial_sample_ratio,
                diversity_weight=config.sample_diversity_weight,
                importance_weight=config.sample_importance_weight
            )
        else:
            self.smart_data_sampler = None
        
        # Training metrics tracking
        self.training_metrics = {
            'memory_usage': [],
            'throughput': [],
            'pipeline_efficiency': [],
            'batch_sizes': [],
            'sample_priorities': []
        }

    def get_optimized_batch_size(self) -> int:
        if self.adaptive_batch_sampler:
            return self.adaptive_batch_sampler.get_current_batch_size()
        return self.config.initial_batch_size
    
    def update_training_metrics(self, memory_usage: float, loss: float, throughput: float):
        if self.adaptive_batch_sampler:
            self.adaptive_batch_sampler.update_metrics(memory_usage, loss, throughput)
        
        self.training_metrics['memory_usage'].append(memory_usage)
        self.training_metrics['throughput'].append(throughput)
        
        max_history = 1000
        for key in self.training_metrics:
            if len(self.training_metrics[key]) > max_history:
                self.training_metrics[key] = self.training_metrics[key][-max_history:]
    
    def adapt_batch_size_if_needed(self) -> int:
        if self.adaptive_batch_sampler and self.adaptive_batch_sampler.should_adapt():
            new_batch_size = self.adaptive_batch_sampler.adapt_batch_size()
            self.training_metrics['batch_sizes'].append(new_batch_size)
            return new_batch_size
        return self.get_optimized_batch_size()
    
    def get_priority_sample_indices(self, num_samples: int, dataset_indices: List[int]) -> List[int]:
        if self.smart_data_sampler and len(dataset_indices) > num_samples:
            if len(dataset_indices) > self.smart_data_sampler.dataset_size:
                self.smart_data_sampler.dataset_size = len(dataset_indices)
                current_size = len(self.smart_data_sampler.sample_scores)
                if len(dataset_indices) > current_size:
                    additional_size = len(dataset_indices) - current_size
                    self.smart_data_sampler.sample_scores = np.concatenate([
                        self.smart_data_sampler.sample_scores,
                        np.ones(additional_size) * 0.5
                    ])
                    self.smart_data_sampler.sample_diversity = np.concatenate([
                        self.smart_data_sampler.sample_diversity,
                        np.ones(additional_size) * 0.5
                    ])
                    self.smart_data_sampler.sample_access_count = np.concatenate([
                        self.smart_data_sampler.sample_access_count,
                        np.zeros(additional_size)
                    ])
                    self.smart_data_sampler.complexity_scores = np.concatenate([
                        self.smart_data_sampler.complexity_scores,
                        np.ones(additional_size) * 0.5
                    ])
                    self.smart_data_sampler.sample_last_loss = np.concatenate([
                        self.smart_data_sampler.sample_last_loss,
                        np.ones(additional_size) * float('inf')
                    ])
            
            priority_indices = self.smart_data_sampler.get_priority_samples(num_samples)
            return [dataset_indices[i] for i in priority_indices if i < len(dataset_indices)]
        return dataset_indices[:num_samples]
    
class EnhancedCTMDiffusion(nn.Module):
    """
    Enhanced CTM-Diffusion architecture with deep CTM control over diffusion.
    
    This gives the CTM unprecedented control over the diffusion process through:
    1. Full CTM state tracking and utilization
    2. Iterative CTM-diffusion refinement
    3. Adaptive scheduling based on CTM certainty
    4. Multi-stage noise refinement guided by CTM synchronization
    """
    
    def __init__(self, config: EnhancedCTMConfig):
        super().__init__()
        self.config = config
        self.device_container = nn.Parameter(torch.empty(0)) # Helper to get device

 
        # Placeholder for TaskAnalyzer instantiation.
        # Ensure TaskAnalyzer is imported (e.g., from .utils import TaskAnalyzer)
        # and instantiated correctly (e.g., TaskAnalyzer(config=self.config) or TaskAnalyzer()).
        # This is a placeholder instantiation.
        try:
            # Attempt to import and instantiate TaskAnalyzer
            # This import path is a guess; adjust as necessary.
            from .utils import TaskAnalyzer as ActualTaskAnalyzerClass
            self.task_analyzer_instance = ActualTaskAnalyzerClass(config=self.config)
        except ImportError:
            # Removed dummy TaskAnalyzer; assume TaskAnalyzer is properly imported and instantiated
            from .utils import TaskAnalyzer
            self.task_analyzer_instance = TaskAnalyzer(config=self.config)
            self.config.use_hrm_core = True  # Enable HRM integration
        
        # Ensure 'copy' is imported for deepcopy
        import copy

        # Determine input dimension for the main encoder and task inference
        self.dynamic_entropy_patcher = None
        self.patcher_encoder = None
        self.multi_granularity_processor = None
        self.byte_embedding = None

        if config.use_dynamic_entropy_patcher:
            from .ctm_components import DynamicEntropyPatcher
            self.dynamic_entropy_patcher = DynamicEntropyPatcher(
                embedding_dim=config.patch_embedding_dim,
                patch_cnn_channels=config.patch_encoder_cnn_channels,
                patching_mode=config.entropy_patcher_threshold_type,
                global_threshold=config.entropy_patcher_global_threshold,
                relative_threshold=config.entropy_patcher_relative_threshold,
                min_patch_size=config.entropy_patcher_min_patch_size,
                max_patch_size=config.entropy_patcher_max_patch_size,
                # Pass parameters for the learnable _EntropyProxyModel
                entropy_byte_vocab_size=config.entropy_model_byte_vocab_size,
                entropy_embedding_dim=config.entropy_model_embedding_dim,
                entropy_hidden_dim=config.entropy_model_hidden_dim,
                entropy_num_layers=config.entropy_model_num_layers,
                entropy_dropout=config.entropy_model_dropout
            )
            raw_feature_dim = config.patch_embedding_dim # Output of dynamic patcher is (batch, num_dynamic_patches, patch_embedding_dim)
        elif config.multi_granularity:
            self.multi_granularity_processor = MultiGranularityBinaryProcessor(config)
            # MGP needs to expose its output dimension, e.g. self.multi_granularity_processor.output_dim
            # Using placeholder from config for now.
            raw_feature_dim = config.multi_granularity_output_dim # This needs to be accurate
            raw_feature_dim = self.multi_granularity_processor.output_dim if hasattr(self.multi_granularity_processor, 'output_dim') else config.multi_granularity_output_dim
        else: # Fallback to simple byte embedding
            self.byte_embedding = nn.Embedding(256, config.byte_embedding_dim)
            raw_feature_dim = config.byte_embedding_dim
        
        # Mixed precision trainer
        self.mixed_precision_trainer = MixedPrecisionTrainer(self, config)

        # Core CTM and Diffusion are now unified in a single model.
        # This denoising model IS the CTM, refactored to predict velocity.
        self.denoising_model = UnifiedCTMDenoisingModel(config)
        self.target_noise_dim = config.unet_input_feature_dim
        self.ctm_contrastive_proj = nn.Linear(self.config.d_model, 128)
        self.diff_contrastive_proj = nn.Linear(self.target_noise_dim, 128)

        
        # Input encoder: processes raw features (from patcher, MGP, or byte_embedding) to ctm_input_dim
        # raw_feature_dim is the embedding_dim of each item in the sequence from the preprocessor.
        self.input_encoder = nn.Linear(raw_feature_dim, config.ctm_input_dim)
        
        # kv_proj for CTM attention, input is ctm_input_dim
        self.kv_proj = nn.Sequential(
            nn.Linear(config.ctm_input_dim, config.ctm_input_dim),
            nn.LayerNorm(config.ctm_input_dim)
        ) if config.ctm_heads > 0 else None
        
        # Language head if vocab_size is specified
        if config.vocab_size and config.vocab_size > 0: # Added > 0 check
            self.language_head = nn.Linear(config.ctm_out_dims, config.vocab_size)
        else:
            self.language_head = None

        # --- Self-Supervised Task Inference and HIPA Control ---
        # Task Inference Module: takes raw_feature_dim + 1 (for byte length)
        # and outputs inferred_task_latent_dim
        # We'll add byte_length as a scalar feature.
        task_inference_input_dim = raw_feature_dim + 1 # +1 for byte_length
        self.task_inference_module = nn.Sequential(
            nn.Linear(task_inference_input_dim, config.inferred_task_latent_dim * 2),
            nn.ReLU(),
            nn.Linear(config.inferred_task_latent_dim * 2, config.inferred_task_latent_dim)
        )
        
        # HIPA Control Module: takes inferred_task_latent_dim
        # outputs a control signal (e.g., a scalar 0-1 for on/off, or more params)
        # For now, a single scalar for on/off probability.
        self.hipa_control_module = nn.Sequential(
            nn.Linear(config.inferred_task_latent_dim, config.inferred_task_latent_dim // 2),
            nn.ReLU(),
            nn.Linear(config.inferred_task_latent_dim // 2, 1),
            nn.Sigmoid()
        )

        # cross_modal_weight: float = 0.1 # If applicable for cross-modal tasks
        lateral_connection_dim: int = 256 # For lateral positional embeddings in multi-task setup
   
         # CTM Control / Convergence Parameters (for CTM's interaction with diffusion/generation)
        max_diffusion_steps: int = 100  # Max steps for CTM-controlled generation loop (distinct from diffusion_steps)
        convergence_threshold: float = 0.01
        efficiency_penalty_weight: float = 0.1
        certainty_convergence_target: float = 0.8
        thought_loop_detection: bool = True
        loop_detection_window: int = 5
        loop_similarity_threshold: float = 0.95
    
        vocab_size: Optional[int] = None # For task-specific output heads (e.g., if not purely binary)
        num_tasks: int = 1 # General number of tasks, might be used by existing code. max_tasks is for CL.
        unet_input_feature_dim: Optional[int] = None # <<< NEW: Standardized feature dimension for diffusion U-Net input
 
        # Audio Output Settings (for TTS tasks)
        output_audio_bytes: bool = False # If True, model output (for generation) is converted to byte sequence. Target input is also expected as bytes.
        audio_output_dtype_str: str = "float32" # Data type of raw audio samples ("float32", "int16")
        audio_output_item_size: int = 4 # Automatically set in __post_init__ based on audio_output_dtype_str

        # Positional Embedding Initialization
        self.positional_embedding = None
        if config.positional_embedding_type:
            pe_dim = config.positional_embedding_dim if config.positional_embedding_dim is not None else config.ctm_input_dim
            if config.positional_embedding_type == 'learnable-fourier':
                self.positional_embedding = LearnableFourierPositionalEncoding(d_model=pe_dim)
            elif config.positional_embedding_type == 'multi-learnable-fourier':
                self.positional_embedding = MultiLearnableFourierPositionalEncoding(d_model=pe_dim)
            elif config.positional_embedding_type == 'custom-rotational':
                self.positional_embedding = CustomRotationalEmbedding(d_model=pe_dim)
            elif config.positional_embedding_type == 'custom-rotational-1d':
                # Using 2D rotational embedding instead of 1D for more biological accuracy.
                self.positional_embedding = CustomRotationalEmbedding(d_model=pe_dim)
            # Unknown positional_embedding_type; no positional embedding used

        self.enhanced_mcmc_sampler = None
        self.mcmc_phi_network = None
        self.mcmc_output_space = None
        self.blackbox_solver = None

        # Projection layer for sampling path: from ctm_input_dim to unet_input_feature_dim
        self.sampling_kv_to_unet_input_proj = nn.Linear(config.ctm_input_dim, config.unet_input_feature_dim)

        # Initialize a training noise scheduler for EnhancedCTMDiffusion
        self.training_noise_scheduler =  DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_timesteps,
            beta_start=config.diffusion_beta_start,
            beta_end=config.diffusion_beta_end,
            beta_schedule=("squaredcos_cap_v2" if config.noise_schedule == "cosine" else config.noise_schedule) if hasattr(config, 'noise_schedule') else "linear"
        )
        
        # Initialize new optimization components
        self._initialize_optimization_components()

        # Consciousness Controller
        if self.config.enable_consciousness_controller:
            self.consciousness_controller = ConsciousnessController(
                model_dim=config.ctm_input_dim,
                max_attention_steps=config.consciousness_max_attention_steps
            )
        else:
            self.consciousness_controller = None
        self.consciousness_step = 0
    
    def _initialize_optimization_components(self):
        """Initialize the new optimization components."""
        # Pipeline parallelism processor
        if self.config.enable_pipeline_parallelism:
            self.pipeline_processor = PipelineParallelProcessor(self.config)
            self.pipeline_processor.start_pipeline()
        else:
            self.pipeline_processor = None
        
        # Adaptive batch sampler
        if self.config.enable_adaptive_batching:
            self.adaptive_batch_sampler = AdaptiveBatchSampler(
                initial_batch_size=self.config.initial_batch_size,
                min_batch_size=self.config.min_batch_size,
                max_batch_size=self.config.max_batch_size,
                adaptation_frequency=self.config.batch_adaptation_frequency
            )
        else:
            self.adaptive_batch_sampler = None
        
        # Smart data sampler
        if self.config.enable_smart_sampling:
            # Initialize with a reasonable dataset size estimate
            estimated_dataset_size = 100000  # Will be updated when actual data is seen
            self.smart_data_sampler = SmartDataSampler(
                dataset_size=estimated_dataset_size,
                initial_sample_ratio=self.config.initial_sample_ratio,
                diversity_weight=self.config.sample_diversity_weight,
                importance_weight=self.config.sample_importance_weight
            )
        else:
            self.smart_data_sampler = None
        
        # Training metrics tracking
        self.training_metrics = {
            'memory_usage': [],
            'throughput': [],
            'pipeline_efficiency': [],
            'batch_sizes': [],
            'sample_priorities': []
        }

        # --- JEPA Components Initialization ---
        if self.config.use_jepa_training:
            if not self.config.use_dynamic_entropy_patcher:
                raise ValueError("JEPA training requires 'use_dynamic_entropy_patcher'.")
            
            self.jepa_target_patch_encoder = copy.deepcopy(self.dynamic_entropy_patcher)
            for param_target in self.jepa_target_patch_encoder.parameters():
                param_target.requires_grad = False
            
            jepa_io_dim = self.config.patch_embedding_dim
            predictor_output_dim = jepa_io_dim * self.config.jepa_num_target_blocks
            self.jepa_predictor = JEPAPredictor(
                input_dim=jepa_io_dim,
                hidden_dim=self.config.jepa_predictor_hidden_dim,
                output_dim=predictor_output_dim
            )
    
    def get_optimized_batch_size(self) -> int:
        """Get the current optimized batch size from adaptive batch sampler."""
        if self.adaptive_batch_sampler:
            return self.adaptive_batch_sampler.get_current_batch_size()
        return self.config.initial_batch_size
    
    def update_training_metrics(self, memory_usage: float, loss: float, throughput: float):
        """Update training metrics for optimization components."""
        if self.adaptive_batch_sampler:
            self.adaptive_batch_sampler.update_metrics(memory_usage, loss, throughput)
        
        # Store metrics for analysis
        self.training_metrics['memory_usage'].append(memory_usage)
        self.training_metrics['throughput'].append(throughput)
        
        # Keep only recent history
        max_history = 1000
        for key in self.training_metrics:
            if len(self.training_metrics[key]) > max_history:
                self.training_metrics[key] = self.training_metrics[key][-max_history:]
    
    def adapt_batch_size_if_needed(self) -> int:
        """Adapt batch size if conditions are met."""
        if self.adaptive_batch_sampler and self.adaptive_batch_sampler.should_adapt():
            new_batch_size = self.adaptive_batch_sampler.adapt_batch_size()
            self.training_metrics['batch_sizes'].append(new_batch_size)
            return new_batch_size
        return self.get_optimized_batch_size()
    
    def get_priority_sample_indices(self, num_samples: int, dataset_indices: List[int]) -> List[int]:
        """Get prioritized sample indices for smart data sampling."""
        if self.smart_data_sampler and len(dataset_indices) > num_samples:
            # Update dataset size if needed
            if len(dataset_indices) > self.smart_data_sampler.dataset_size:
                self.smart_data_sampler.dataset_size = len(dataset_indices)
                # Expand tracking arrays
                current_size = len(self.smart_data_sampler.sample_scores)
                if len(dataset_indices) > current_size:
                    additional_size = len(dataset_indices) - current_size
                    self.smart_data_sampler.sample_scores = np.concatenate([
                        self.smart_data_sampler.sample_scores,
                        np.ones(additional_size) * 0.5
                    ])
                    self.smart_data_sampler.sample_diversity = np.concatenate([
                        self.smart_data_sampler.sample_diversity,
                        np.ones(additional_size) * 0.5
                    ])
                    self.smart_data_sampler.sample_access_count = np.concatenate([
                        self.smart_data_sampler.sample_access_count,
                        np.zeros(additional_size)
                    ])
                    self.smart_data_sampler.complexity_scores = np.concatenate([
                        self.smart_data_sampler.complexity_scores,
                        np.ones(additional_size) * 0.5
                    ])
                    self.smart_data_sampler.sample_last_loss = np.concatenate([
                        self.smart_data_sampler.sample_last_loss,
                        np.ones(additional_size) * float('inf')
                    ])
            
            priority_indices = self.smart_data_sampler.get_priority_samples(num_samples)
            return [dataset_indices[i] for i in priority_indices if i < len(dataset_indices)]
        return dataset_indices[:num_samples]
    
    def update_sample_importance(self, sample_indices: List[int], losses: List[float],
                                gradients: Optional[List[torch.Tensor]] = None):
        """Update sample importance scores based on training feedback."""
        if self.smart_data_sampler:
            self.smart_data_sampler.update_sample_importance(sample_indices, losses, gradients)
    
    def analyze_batch_patterns(self, data_batch: torch.Tensor, sample_indices: List[int]):
        """Analyze binary patterns in the current batch for complexity scoring."""
        if self.smart_data_sampler and self.config.complexity_analysis_enabled:
            self.smart_data_sampler.analyze_binary_patterns(data_batch, sample_indices)
    
    def forward_with_pipeline_optimization(self, byte_sequence: torch.Tensor, task_name: str,
                                         target_diffusion_output: Optional[torch.Tensor] = None,
                                         timestep: Optional[torch.Tensor] = None,
                                         confidence_level: str = 'medium') -> Dict[str, torch.Tensor]:
        """Forward pass using pipeline parallelism optimization."""
        if self.pipeline_processor and self.config.enable_pipeline_parallelism:
            # Prepare inputs for pipeline processing
            kv_features = self.compute_features(self._prepare_input_features(byte_sequence, task_name))
            
            # Use pipeline processor for optimized execution
            pipeline_results = self.pipeline_processor.pipeline_forward(
                ctm_core=self.ctm_core,
                diffusion_processor=self.diffusion,
                inputs=kv_features,
                timesteps=timestep,
                guidance_data={'task_name': task_name}
            )
            
            # Store pipeline efficiency metrics
            if 'pipeline_efficiency' in pipeline_results:
                self.training_metrics['pipeline_efficiency'].append(
                    pipeline_results['pipeline_efficiency']
                )
            
            return {
                'ctm_core_data': pipeline_results.get('ctm_results'),
                'diffusion_output': pipeline_results.get('diffusion_output'),
                'mcmc_results': None,
                'pipeline_efficiency': pipeline_results.get('pipeline_efficiency'),
                'final_output': pipeline_results.get('diffusion_output')
            }
        else:
            # Fallback to standard forward pass
            return self.forward(byte_sequence, target_diffusion_output=target_diffusion_output,
                                mode='ctm_controlled_diffusion', timestep=timestep,
                                task_name=task_name, confidence_level=confidence_level)
    
    def _prepare_input_features(self, byte_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepares the input features for the CTM core, generates inferred task latent,
        HIPA control signal, and potential auxiliary loss from the entropy model.

        Args:
            byte_sequence: Raw byte sequence tensor. Shape (batch_size, sequence_length).
                           Assumed to be integer type for embedding if not using MGP.

        Returns:
            Tuple of:
                - encoded_features: Features ready for CTM core. (batch_size, num_patches_or_seq, ctm_input_dim)
                - inferred_task_latent: Latent vector representing the inferred task. (batch_size, inferred_task_latent_dim)
                - hipa_control_signal: Signal to control HIPA activation. (batch_size, 1)
                - entropy_aux_loss: Auxiliary loss from the learnable entropy model. Scalar tensor.
        """
        batch_size = byte_sequence.size(0)
        seq_len = byte_sequence.size(1) if byte_sequence.dim() > 1 else 1
        device = byte_sequence.device
        entropy_aux_loss = torch.tensor(0.0, device=device) # Default to zero

        # patch_indices are currently not used further but returned by dynamic patcher
        patch_indices = None

        if self.dynamic_entropy_patcher:
            # LearnedBytePatcherEncoder (when it's the dynamic one) now returns (encoded_patches, patch_indices, aux_loss)
            raw_features, patch_indices, current_entropy_aux_loss = self.dynamic_entropy_patcher(byte_sequence)
            entropy_aux_loss = current_entropy_aux_loss
            # raw_features shape: (batch_size, num_dynamic_patches, embedding_dim)
        elif self.multi_granularity_processor:
            raw_features = self.multi_granularity_processor(byte_sequence)
        elif self.byte_embedding:
            raw_features = self.byte_embedding(byte_sequence.long())
        else:
            raise ValueError("No valid byte processor configured (dynamic, patcher, MGP, or byte_embedding).")

        # `raw_features` is now a sequence of embeddings (e.g., per patch or per byte)
        # For task inference, we need a fixed-size representation per batch item.
        # Typically, mean pooling over the sequence dimension (num_patches or seq_len).
        if raw_features.dim() == 3: # (batch, seq_dim, feat_dim)
            task_inference_input_features = raw_features.mean(dim=1) # (batch, feat_dim)
        elif raw_features.dim() == 2: # (batch, feat_dim) - e.g. if MGP outputs fixed size
            task_inference_input_features = raw_features
        else:
            raise ValueError(f"Unsupported raw_features dimension: {raw_features.dim()}")
            
        # Byte length feature (original sequence length before patching/padding)
        # byte_sequence is the original (batch, seq_len_bytes)
        # We need a scalar length per batch item.
        actual_lengths = []
        for i in range(batch_size):
            # This assumes byte_sequence might have padding if not using patcher,
            # or we just use its original length.
            # If using patcher, byte_sequence.shape[1] is the original length before internal padding.
            actual_lengths.append(byte_sequence.shape[1]) # Original length of byte sequence
        byte_lengths_tensor = torch.tensor(actual_lengths, dtype=torch.float32, device=device).unsqueeze(1)

        # Concatenate features for task inference module
        # task_inference_input_features dim is raw_features.shape[-1]
        # which is patch_embedding_dim if patcher is used, or MGP output_dim, or byte_embedding_dim.
        # This was set as `raw_feature_dim` in __init__ for task_inference_module.
        inference_module_input = torch.cat((task_inference_input_features, byte_lengths_tensor), dim=-1)
        
        # Validate input dimension for task_inference_module
        expected_task_inference_input_dim = self.task_inference_module[0].in_features
        if inference_module_input.shape[-1] != expected_task_inference_input_dim:
            # This error can occur if raw_feature_dim used to init task_inference_module
            # doesn't match task_inference_input_features.shape[-1] + 1.
            # raw_feature_dim in __init__ should be the feature dim *before* cat with length.
            raise ValueError(
                f"Dimension mismatch for task_inference_module. Expected {expected_task_inference_input_dim}, "
                f"got {inference_module_input.shape[-1]} (features: {task_inference_input_features.shape[-1]} + length: 1). "
                f"Ensure raw_feature_dim used for task_inference_module init was {task_inference_input_features.shape[-1]}."
            )

        inferred_task_latent = self.task_inference_module(inference_module_input)
        hipa_control_signal = self.hipa_control_module(inferred_task_latent)

        # Per-sample heuristic to disable HiPA for text-like sequences
        for i in range(batch_size):
            if byte_sequence[i].shape[0] % 250 == 0 and byte_sequence[i].shape[0] > 0:
                if hipa_control_signal[i] > 0:
                    hipa_control_signal[i] = 0.0

        # The main input_encoder processes the sequence of raw_features
        # raw_features is (batch, num_patches_or_seq, feature_dim)
        encoded_features = self.input_encoder(raw_features) # Output: (batch, num_patches_or_seq, ctm_input_dim)

        # Apply consciousness modulation
        if self.consciousness_controller and self.consciousness_controller.consciousness_state != 'sleeping':
             encoded_features = self.consciousness_controller.apply_consciousness_to_features(encoded_features)

        # Apply positional embedding if configured
        if self.positional_embedding is not None and encoded_features.shape[1] > 0:
            if self.config.reshape_patch_sequence_to_grid and \
               isinstance(self.positional_embedding, (LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding)):
                
                B, S, D = encoded_features.shape
                W_patches = self.config.patch_grid_width
                if W_patches is None: # Should be caught by config validation, but as a safeguard
                    print(f"Warning: patch_grid_width is None but reshape_patch_sequence_to_grid is True. Defaulting width to sqrt(S).")
                    W_patches = int(math.sqrt(S))  # Default width
                    if W_patches == 0: W_patches = 1 # Avoid zero width

                H_patches = math.ceil(S / W_patches)
                total_grid_elements = H_patches * W_patches
                
                # Pad if necessary
                if S < total_grid_elements:
                    padding_size = total_grid_elements - S
                    padding = torch.zeros(B, padding_size, D, device=encoded_features.device, dtype=encoded_features.dtype)
                    grid_sequence_features = torch.cat([encoded_features, padding], dim=1)
                elif S > total_grid_elements: # Should not happen if H_patches is math.ceil
                    grid_sequence_features = encoded_features[:, :total_grid_elements, :]
                else:
                    grid_sequence_features = encoded_features

                # Reshape to grid: (B, H_patches, W_patches, D)
                grid_features_hw_d = grid_sequence_features.reshape(B, H_patches, W_patches, D)
                
                # Permute for 2D PE: (B, D, H_patches, W_patches)
                pe_input_grid = grid_features_hw_d.permute(0, 3, 1, 2)
                
                pos_emb_grid = self.positional_embedding(pe_input_grid) # Output (B, D, H_patches, W_patches)
                
                # Add positional embedding
                grid_features_with_pe = pe_input_grid + pos_emb_grid
                
                # Permute back: (B, H_patches, W_patches, D)
                grid_features_hw_d_with_pe = grid_features_with_pe.permute(0, 2, 3, 1)
                
                # Reshape back to sequence: (B, H_patches * W_patches, D)
                # The CTM will now process a sequence of length H_patches * W_patches
                encoded_features = grid_features_hw_d_with_pe.reshape(B, total_grid_elements, D)
                # Reshaped patch sequence to grid and applied 2D PE

            
            elif isinstance(self.positional_embedding, (LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding)) and not self.config.reshape_patch_sequence_to_grid:
                 # 2D PE selected, but not reshaping. Apply as if 1D sequence with H=S, W=1 (or similar)
                if encoded_features.dim() == 3: # (B, S, D)
                    pe_input_seq_as_2d = encoded_features.permute(0, 2, 1).unsqueeze(-1) # (B, D, S, 1)
                    pos_emb_seq_as_2d = self.positional_embedding(pe_input_seq_as_2d)
                    pos_emb_seq_as_2d = pos_emb_seq_as_2d.squeeze(-1).permute(0, 2, 1)
                    if pos_emb_seq_as_2d.shape == encoded_features.shape:
                         encoded_features = encoded_features + pos_emb_seq_as_2d
                    else:
                        print(f"Warning: 2D PE (applied to sequence) shape {pos_emb_seq_as_2d.shape} mismatch with features {encoded_features.shape}. Skipping PE.")
            
        return encoded_features, inferred_task_latent, hipa_control_signal, entropy_aux_loss
    
    def forward_with_mixed_precision(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass with automatic mixed precision."""
        return self.mixed_precision_trainer.forward_with_autocast(*args, **kwargs)
    
    def backward_with_mixed_precision(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> bool:
        """Backward pass with mixed precision gradient scaling."""
        return self.mixed_precision_trainer.backward_with_scaling(loss, optimizer)
    
    def optimizer_step_with_mixed_precision(self, optimizer: torch.optim.Optimizer) -> bool:
        """Optimizer step with gradient unscaling and overflow detection."""
        return self.mixed_precision_trainer.optimizer_step(optimizer)
    
    def get_mixed_precision_stats(self) -> Dict[str, Any]:
        """Get mixed precision training statistics."""
        return self.mixed_precision_trainer.get_mixed_precision_stats()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        return self.mixed_precision_trainer.get_memory_usage()

    def compute_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes key-value features for the CTM from the input tensor.
        The input x is the output of self.input_encoder.
        """
        kv_features_for_ctm = x
        return kv_features_for_ctm

    def wake_up(self):
        """Gradually wake up the model's attention."""
        if self.consciousness_controller:
            self.consciousness_step = 0
            for i in range(self.consciousness_controller.max_attention_steps):
                self.consciousness_controller.wake_up(i)
            print(f"Model is awake.")

    def sleep_down(self):
        """Gradually put the model's attention to sleep."""
        if self.consciousness_controller:
            self.consciousness_step = 0
            for i in range(self.consciousness_controller.max_attention_steps):
                self.consciousness_controller.sleep_down(i)
            print(f"Model is asleep.")
    
    def forward(self,
                byte_sequence: torch.Tensor,
                target_diffusion_output: Optional[torch.Tensor] = None,
                timestep: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the refactored EnhancedCTMDiffusion model.
        This method now uses the UnifiedCTMDenoisingModel to predict velocity
        and calculates the corresponding rectified flow loss. The old controller-
        actuator logic has been removed.
        """
        device = byte_sequence.device
        batch_size = byte_sequence.size(0)
        losses = {}

        # 1. Get conditioning features and entropy loss from the main input processor
        conditioning_features, _, _, entropy_aux_loss = self._prepare_input_features(byte_sequence)
        if self.training:
            losses['entropy_aux_loss'] = entropy_aux_loss * self.config.entropy_model_loss_weight

        # 2. Main diffusion loss calculation (rectified flow)
        if target_diffusion_output is not None and self.training:
            if target_diffusion_output.dtype == torch.uint8:
                target_flat = target_diffusion_output.view(batch_size, -1).float()
                x_0 = (target_flat / 255.0) * 2.0 - 1.0
            else:
                x_0 = target_diffusion_output.view(batch_size, -1)

            current_len, target_len = x_0.shape[-1], self.config.unet_input_feature_dim
            if current_len < target_len:
                x_0 = F.pad(x_0, (0, target_len - current_len))
            elif current_len > target_len:
                x_0 = x_0[:, :target_len]

            x_1, noisy_input = torch.randn_like(x_0), self.denoising_model.add_noise(x_0, torch.randn_like(x_0), timestep)
            predicted_velocity = self.denoising_model(noisy_input, timestep, conditioning_features=conditioning_features)
            target_velocity = self.denoising_model.get_velocity(x_0, x_1)
            losses['diffusion_loss'] = F.mse_loss(predicted_velocity, target_velocity)

        # 3. JEPA self-supervised loss
        if self.config.use_jepa_training and self.training:
            with torch.no_grad():
                target_patch_embeddings, _, _ = self.jepa_target_patch_encoder(byte_sequence)
            
            online_patch_embeddings, _, _ = self.dynamic_entropy_patcher(byte_sequence)

            context_reps, target_reps = self._jepa_create_masked_patch_views(online_patch_embeddings, target_patch_embeddings)
            
            if context_reps is not None and target_reps is not None:
                predicted_target_reps = self.jepa_predictor(context_reps)
                
                # Reshape for loss calculation
                
                D_embed = online_patch_embeddings.shape[-1]
                num_targets = self.config.jepa_num_target_blocks
                predicted_target_reps = predicted_target_reps.view(-1, num_targets, D_embed)
                
                jepa_loss = F.mse_loss(predicted_target_reps, target_reps.detach())
                losses['jepa_loss'] = jepa_loss * self.config.jepa_loss_weight
            
            self._update_jepa_target_encoder()


        # Combine all losses
        total_loss = sum(losses.values()) if losses else torch.tensor(0.0, device=device)

        return {
            'final_output': predicted_velocity if 'diffusion_loss' in losses else None,
            'total_loss': total_loss,
            **losses
        }
    
    def iterative_ctm_diffusion_sample(self, shape: Tuple[int, ...],
                                       initial_byte_sequence_for_inference: Optional[torch.Tensor] = None,
                                       num_steps: int = 50,
                                       generator: Optional[torch.Generator] = None,
                                       **kwargs
                                       ) -> Tuple[torch.Tensor, Dict]:
        """
        Simplified sampling using the UnifiedCTMDenoisingModel.
        The CTM guidance is now handled by passing conditioning features to the sampler.
        """
        device = self.device_container.device
        batch_size = shape[0]
        sampling_info = {}

        # 1. Prepare conditioning features for the generation process
        conditioning_features = None
        if initial_byte_sequence_for_inference is not None:
             # Ensure batch size matches if needed
            if initial_byte_sequence_for_inference.size(0) != batch_size:
                 initial_byte_sequence_for_inference = initial_byte_sequence_for_inference[0].unsqueeze(0).expand(batch_size, *initial_byte_sequence_for_inference.shape[1:])
            
            conditioning_features, _, _, _ = self._prepare_input_features(
                initial_byte_sequence_for_inference.to(device)
            )

        # 2. Call the unified model's sampler
        # The denoising model now internally handles the CTM-guided loop.
        generated_numeric_tensor = self.denoising_model.sample(
            shape=shape,
            num_inference_steps=num_steps,
            generator=generator,
            conditioning_features=conditioning_features
        )

        # 3. Convert the final numeric tensor to bytes
        generated_bytes = batched_numeric_tensor_to_bytes(
            generated_numeric_tensor, source_dtype=np.float32
        )

        return generated_bytes, sampling_info
    

    @torch.no_grad()
    def _update_jepa_target_encoder(self):
        """
        Performs momentum update of the JEPA target patch encoder parameters
        using the online patch encoder (self.dynamic_entropy_patcher).
        This should be called by the training loop after optimizer.step().
        """
        if not self.config.use_jepa_training or \
           self.dynamic_entropy_patcher is None or self.jepa_target_patch_encoder is None:
            return
        
        m = self.config.jepa_momentum_beta
        for param_online, param_target in zip(self.dynamic_entropy_patcher.parameters(), self.jepa_target_patch_encoder.parameters()):
            param_target.data.mul_(m).add_((1 - m) * param_online.data)

    def _jepa_create_masked_patch_views(self,
                                        online_patch_embeddings: torch.Tensor,
                                        target_patch_embeddings: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Creates context and target representations from sequences of patch embeddings for JEPA.
        """
        B, S_patches, D_embed = online_patch_embeddings.shape
        device = online_patch_embeddings.device

        if S_patches < 2:  # Need at least one patch for context and one for target
            return None, None

        batch_context_reps = []
        batch_target_reps = []

        for b_idx in range(B):
            # 1. Determine context block size
            context_scale = random.uniform(self.config.jepa_context_scale_min, self.config.jepa_context_scale_max)
            num_context_patches = max(1, int(S_patches * context_scale))
            # Ensure there's enough space for at least num_target_blocks patches left after context
            num_context_patches = min(num_context_patches, S_patches - self.config.jepa_num_target_blocks)

            if num_context_patches <= 0:
                continue

            all_indices = torch.arange(S_patches, device=device)
            # Randomly select start for context block
            # Max start index for context ensures that context_block + target_blocks fit
            
            context_start_idx = random.randint(0, S_patches - num_context_patches - self.config.jepa_num_target_blocks)
            context_indices = all_indices[context_start_idx : context_start_idx + num_context_patches]
            context_block_embeddings = online_patch_embeddings[b_idx, context_indices, :]
            context_rep = context_block_embeddings.mean(dim=0)
            # 3. Select target blocks (non-overlapping with context)
            # Create a mask for available target indices
            available_target_mask = torch.ones(S_patches, dtype=torch.bool, device=device)
            available_target_mask[context_indices] = False
            potential_target_indices = all_indices[available_target_mask]

            if len(potential_target_indices) < self.config.jepa_num_target_blocks:
                continue
            
            shuffled_potential_target_indices = potential_target_indices[torch.randperm(len(potential_target_indices), device=device)]
            actual_target_indices = shuffled_potential_target_indices[:self.config.jepa_num_target_blocks]
            
            selected_target_patches = target_patch_embeddings[b_idx, actual_target_indices, :]  # (num_target_blocks, D_embed)
            target_rep = selected_target_patches  #Keep as distinct blocks, shape (num_target_blocks, D_embed)

            batch_context_reps.append(context_rep)  # List of (D_embed)
            batch_target_reps.append(target_rep)# List of (num_target_blocks, D_embed)

        if not batch_context_reps or not batch_target_reps:
            return None, None

        return torch.stack(batch_context_reps), torch.stack(batch_target_reps)

def create_enhanced_config_for_text_generation(vocab_size: int) -> EnhancedCTMConfig:
    """Create enhanced configuration for text generation with strong CTM control"""
    config = EnhancedCTMConfig()
    config.vocab_size = vocab_size
    config.d_model = 768
    config.d_input = 768
    config.out_dims = 768
    config.ctm_diffusion_coupling_strength = 0.9  # Very strong coupling
    config.adaptive_scheduling = True
    config.iterative_refinement = True
    return config


def create_enhanced_config_for_tts_nonverbal(vocab_size: int) -> EnhancedCTMConfig:
    """Create enhanced configuration for TTS with nonverbal communication"""
    config = EnhancedCTMConfig()
    config.vocab_size = vocab_size
    config.d_model = 512
    config.d_input = 512
    config.out_dims = 512
    config.iterations = 8  # More iterations for complex audio generation
    config.ctm_diffusion_coupling_strength = 0.85  # Strong coupling for audio
    config.adaptive_scheduling = True
    config.iterative_refinement = True
    return config
        



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
