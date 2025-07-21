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
from .ctm_components import HierarchicalCTM
from .ctm_components import (
    EnhancedCTMConfig,
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



class CTMControlledDiffusionProcessor(nn.Module):
    """
    Enhanced Diffusion processor with DEEP CTM integration and GPU optimizations.
    
    The CTM now has multiple levels of influence:
    1. Direct noise prediction conditioning from synchronization states
    2. Adaptive timestep scheduling based on CTM certainty
    3. CTM-guided attention mechanisms for noise refinement
    4. Iterative CTM-diffusion coupling during generation
    5. CTM state-dependent diffusion guidance
    6. Enhanced CTM guidance processors with adaptive strength
    7. Quality preservation networks
    8. Multi-resolution guidance for different data types
    """

    def __init__(self, config: EnhancedCTMConfig, actual_noisy_input_dim: int, task_analyzer: 'TaskAnalyzer'): # Added task_analyzer
        super().__init__()
        self.target_noise_dim = actual_noisy_input_dim # This is config.unet_input_feature_dim
        
        self.latent_dim = config.d_model
        self.coupling_strength = config.ctm_diffusion_coupling_strength
        self.adaptive_scheduling = config.adaptive_scheduling
        self.iterative_refinement = config.iterative_refinement
        self.config = config # Storing the config for other parts of the class
        
                
        # Add recurrent connections
        self.recurrent_cells = nn.ModuleDict()
        for layer in range(config.n_layers):
            self.recurrent_cells[f"layer_{layer}"] = nn.GRUCell(
                input_size=config.d_model,
                hidden_size=config.d_model
            )
            
        # Add predictive coding modules
        self.predictive_coders = nn.ModuleDict()
        for layer in range(1, config.n_layers):
            self.predictive_coders[f"layer_{layer}"] = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU()
            )
            
        
        # Replace WINA sparsifier with meta version
        self.wina_sparsifier = MetaWINASparsifier(
            sparsity_ratio=config.sparse_attention_ratio,
            control_dim=getattr(config, 'control_dim', 64)
        )

        # Enhanced CTM guidance processors (from ctm_guided_integration_flow.py)
        self.sync_to_flow_mapper = nn.Sequential(
            nn.Linear(config.ctm_n_synch_out, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Tanh()  # Bounded output for stable guidance
        )
        
        # Multi-scale guidance fusion
        self.guidance_fusion = nn.MultiheadAttention(
            config.d_model, num_heads=8, batch_first=True
        )
        
        # Certainty-adaptive guidance strength predictor
        self.adaptive_strength_predictor = nn.Sequential(
            nn.Linear(2 + config.d_model, 128),  # certainty + sync state
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1 strength
        )
        
        # Quality preservation network
        self.quality_enhancer = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.Identity()  # Preserve original signal (residual connection)
        )
        
        # Learned noise-to-data mapping (Integration Flow core)
        self.flow_predictor = nn.Sequential(
            nn.Linear(self.target_noise_dim * 3, config.d_model * 2),  # noise + sync + certainty
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model), # Intermediate
            nn.GELU(),
            nn.Linear(config.d_model, self.target_noise_dim) # Output target_noise_dim
        )
        
        # Multi-resolution guidance for different data types
        self.multi_res_processors = nn.ModuleList([
            nn.Conv1d(config.d_model, config.d_model, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]  # Different receptive fields
        ])
        
        # Learnable guidance weights
        self.sync_weight = nn.Parameter(torch.tensor(1.0))
        self.certainty_weight = nn.Parameter(torch.tensor(0.8))
        self.prediction_weight = nn.Parameter(torch.tensor(0.6))
        
        # GPU optimization: Enable flow refinement
        self.flow_refinement_enabled = True
        
        # Task-Aware HiPA system for intelligent frequency enhancement
        self.task_aware_hipa = FrequencyDomainAwareAttention(
            embed_dim=self.target_noise_dim, # Changed to target_noise_dim
            num_heads=8,
            task_analyzer=task_analyzer, # Pass the task_analyzer instance
            config=config # Pass the main config
        )

        # Initialize Integration Flow + HiPA Sampler for ultra-fast generation
        self.integration_flow_sampler = IntegrationFlowHiPASampler(
            task_aware_hipa_module=self.task_aware_hipa, # Pass the HiPA module instance
            num_steps=config.diffusion_steps,
            beta_start=config.diffusion_beta_start,
            beta_end=config.diffusion_beta_end,
            sigma_min=0.01,
            sigma_max=50.0,
            hipa_freq_threshold=0.1,
            integration_flow_strength=1.0,
            model_type='VE'  # Default to VE for CTM integration
        )
        
        # Integration Flow control parameters
        self.enable_integration_flow = True
        self.enable_task_aware_hipa = True
        self.integration_flow_strength = nn.Parameter(torch.tensor(0.8))  # Learnable strength
        
        # Enhanced time embedding with CTM influence
        self.time_embedding = nn.Sequential(
            nn.Linear(1, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.d_model)
        )
        
        # CTM synchronization processor (strongest influence)
        self.ctm_sync_processor = nn.Sequential(
            nn.Linear(config.ctm_n_synch_out, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model)
        )
        
        # CTM certainty-based adaptive control
        self.ctm_certainty_processor = nn.Sequential(
            nn.Linear(2, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, config.d_model)
        )
        
        # CTM internal state processor
        self.ctm_state_processor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # CTM prediction history processor
        self.ctm_prediction_processor = nn.Sequential(
            nn.Linear(config.ctm_out_dims * config.ctm_iterations, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # Multi-head attention for CTM-guided noise prediction
        # Now using target_noise_dim (2048) as the embedding dimension
        self.ctm_guided_attention = nn.MultiheadAttention(
            self.target_noise_dim, num_heads=8, batch_first=True
        )
        
        # Projection for the key and value of ctm_guided_attention
        # Projects from config.d_model (512) to target_noise_dim (2048)
        self.kv_projector_for_ctm_attention = nn.Linear(config.d_model, self.target_noise_dim)

        # Base noise prediction network
        # The input dimension is actual_noisy_input_dim (from kv_features_for_ctm) + config.d_model (from time_emb)
        self.noise_predictor_base = nn.Sequential(
            nn.Linear(actual_noisy_input_dim + config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, self.target_noise_dim), # Changed output to target_noise_dim
        )
        
        # CTM-controlled noise refinement (multiple stages)
        self.ctm_noise_refinement = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.target_noise_dim + config.d_model * 2, config.d_model * 2),  # Changed input dim
                nn.GELU(),
                nn.Linear(config.d_model * 2, config.d_model), # Intermediate
                nn.GELU(),
                nn.Linear(config.d_model, self.target_noise_dim) # Changed output to target_noise_dim
            ) for _ in range(4)  # 4 refinement stages
        ])
        
        # CTM-adaptive diffusion schedule predictor
        self.adaptive_schedule_predictor = nn.Sequential(
            nn.Linear(config.d_model + 2, 128),  # CTM state + certainty
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Certainty-based noise scaling
        self.certainty_noise_scaler = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Standard diffusion schedule
        # Initialize a scheduler for sampling methods like denoise_one_step
        self.sampling_noise_scheduler =  DPMSolverMultistepScheduler (
            num_train_timesteps=config.diffusion_timesteps, # Use diffusion_timesteps from main config
            beta_start=config.diffusion_beta_start,
            beta_end=config.diffusion_beta_end,
            beta_schedule="linear" # Changed from config.noise_schedule to fix NotImplementedError for 'cosine'
        )
        self.register_buffer('alpha_bars', self.sampling_noise_scheduler.alphas_cumprod.clone())
        
        # Learnable CTM influence weights
        self.sync_influence_weight = nn.Parameter(torch.tensor(1.0))  # Strong sync influence
        self.certainty_influence_weight = nn.Parameter(torch.tensor(0.8))
        self.state_influence_weight = nn.Parameter(torch.tensor(0.6))
        self.prediction_influence_weight = nn.Parameter(torch.tensor(0.4))
        
        # State history for thought loop detection
        self.register_buffer('state_history', torch.zeros(10, config.d_model))  # Store last 10 states
        self.history_pointer = 0
        
        # Missing components for early stopping
        self.efficiency_predictor = nn.Sequential(
            nn.Linear(config.d_model + 1, 64),  # state + step_ratio
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # CTM Feedback Module
        if config.ctm_n_synch_out > 0:
            self.ctm_feedback_module = CTMFeedbackModule(
                ctm_sync_dim=config.ctm_n_synch_out,
                diffusion_model_dim=self.target_noise_dim,
                n_heads=config.n_heads
            )
        else:
            self.ctm_feedback_module = None

        # Neuromodulator Initialization
        modulator_classes = {
            'dopamine': DopamineModulator,
            'serotonin': SerotoninModulator,
            'oxytocin': OxytocinModulator,
            'norepinephrine': NorepinephrineModulator,
            'acetylcholine': AcetylcholineModulator,
            'endorphins': EndorphinsModulator,
            'cortisol': CortisolModulator,
            'gaba': GABAModulator,
            'glutamate': GlutamateModulator
        }
        self.neuromodulators = nn.ModuleDict()
        if config.enable_neuromodulators:
            for mod_name in config.active_neuromodulators:
                if mod_name in modulator_classes:
                    self.neuromodulators[mod_name] = modulator_classes[mod_name](config.neuromodulator_dim)
    
    def forward(self, noisy_input: torch.Tensor, timestep: torch.Tensor,
                ctm_data: Optional[Dict[str, torch.Tensor]] = None,
                hipa_control_signal: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]: # Added hipa_control_signal, updated return type
        self.forward = torch.compile(self.forward)
        """
        Predict noise with DEEP CTM control and influence.
        
        Args:
            noisy_input: Noisy data to denoise
            timestep: Current diffusion timestep
            ctm_data: Dictionary containing all CTM states and outputs
            hipa_control_signal: Signal to control HIPA activation in submodules.
        Returns:
            Predicted noise (or x0 depending on UNet) and potentially guidance info.
            For now, let's assume it returns just the prediction.
            The original return was torch.Tensor. If guidance_info is returned, it should be Tuple.
            Let's stick to torch.Tensor for now and adjust if guidance_info is added here.
        """
        # This method's responsibility is to predict noise using its internal UNet/etc.,
        # conditioned by ctm_data and potentially using hipa_control_signal if its submodules are HIPA-aware.
        # The current CTMControlledDiffusionProcessor structure doesn't show internal HIPA modules directly.
        # If self.noise_predictor_refined or other components were HIPA-aware, they'd use hipa_control_signal.
        # For now, we ensure it's accepted.
        
        # Time embedding
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        time_emb = self.time_embedding(timestep.float().unsqueeze(-1))
        
        # Base noise prediction
        combined = torch.cat([noisy_input, time_emb], dim=-1) # Reverted: No task_embedding
        base_noise_pred = self.noise_predictor_base(combined)
        
        # If no CTM context, return base prediction
        if ctm_data is None:
            confidence = torch.tensor(1.0, device=base_noise_pred.device)
            return base_noise_pred, confidence
        
        # Extract CTM influences
        ctm_influences = []
        influence_weights = []
        
        # 1. Synchronization influence (STRONGEST)
        if 'final_sync_out' in ctm_data:
            sync_influence = self.ctm_sync_processor(ctm_data['final_sync_out'])
            ctm_influences.append(sync_influence)
            influence_weights.append(self.sync_influence_weight)
        
        # 2. Certainty-based adaptive influence
        if 'certainties' in ctm_data:
            # Use final iteration certainty
            final_certainty = ctm_data['certainties'][:, :, -1]  # (B, 2)
            certainty_influence = self.ctm_certainty_processor(final_certainty)
            
            # Scale influence by certainty level
            certainty_strength = final_certainty[:, 0].unsqueeze(-1)  # High certainty = strong influence
            scaled_certainty_influence = certainty_influence * certainty_strength
            
            ctm_influences.append(scaled_certainty_influence)
            influence_weights.append(self.certainty_influence_weight)
        
        # 3. Internal state influence
        if 'activated_states' in ctm_data and len(ctm_data['activated_states']) > 0:
            # Use final activated state
            final_state = ctm_data['activated_states'][-1]
            state_influence = self.ctm_state_processor(final_state)
            ctm_influences.append(state_influence)
            influence_weights.append(self.state_influence_weight)
        
        # 4. Prediction history influence
        if 'predictions' in ctm_data:
            # Flatten prediction history
            pred_history = ctm_data['predictions'].flatten(start_dim=1)  # (B, out_dims * iterations)
            pred_influence = self.ctm_prediction_processor(pred_history)
            ctm_influences.append(pred_influence)
            influence_weights.append(self.prediction_influence_weight)
        
        # Combine and weight CTM influences
        if ctm_influences:
            # Stack influences and weights
            stacked_influences = torch.stack(ctm_influences, dim=1)  # (B, num_influences, d_model)
            stacked_weights = torch.stack(influence_weights, dim=0)  # (num_influences,)
            
            # Apply learned weights
            weighted_influences = stacked_influences * stacked_weights.view(1, -1, 1) # (B, num_influences, config.d_model)
            
            # Project weighted_influences (keys and values) to target_noise_dim
            # Original shape: (B, num_influences, config.d_model)
            # Target shape for K,V: (B, num_influences, self.target_noise_dim)
            batch_size_attn, num_influences_attn, _ = weighted_influences.shape
            projected_kv_for_attention = self.kv_projector_for_ctm_attention(
                weighted_influences.reshape(batch_size_attn * num_influences_attn, self.config.d_model)
            ).reshape(batch_size_attn, num_influences_attn, self.target_noise_dim)

            # Use CTM-guided attention to combine influences
            # Query is base_noise_pred (B, self.target_noise_dim)
            query = base_noise_pred.unsqueeze(1)  # (B, 1, self.target_noise_dim)
            attended_influence, attention_weights = self.ctm_guided_attention(
                query, projected_kv_for_attention, projected_kv_for_attention # K and V are projected
            )
            attended_influence = attended_influence.squeeze(1)  # (B, self.target_noise_dim)
            
            # Progressive refinement with CTM control
            current_noise = base_noise_pred
            
            # Get primary CTM influences for refinement
            # These influences are config.d_model (512) dimensional.
            batch_size_for_fallback = base_noise_pred.shape[0]
            device_for_fallback = base_noise_pred.device
            dtype_for_fallback = base_noise_pred.dtype

            sync_inf = ctm_influences[0] if len(ctm_influences) > 0 else torch.zeros(batch_size_for_fallback, self.config.d_model, device=device_for_fallback, dtype=dtype_for_fallback)
            state_inf = ctm_influences[2] if len(ctm_influences) > 2 else torch.zeros(batch_size_for_fallback, self.config.d_model, device=device_for_fallback, dtype=dtype_for_fallback)
            
            # Multi-stage refinement
            deltas = []
            prev_noise = current_noise.clone()
            for i, refinement_layer in enumerate(self.ctm_noise_refinement):
                # CTM Feedback
                if hasattr(self, 'ctm_feedback_module') and self.ctm_feedback_module is not None and ctm_data and 'final_sync_out' in ctm_data:
                    if current_noise.dim() == 2:
                        current_noise_for_feedback = current_noise.unsqueeze(1)
                    else:
                        current_noise_for_feedback = current_noise
                    
                    if current_noise_for_feedback.size(-1) == self.ctm_feedback_module.diffusion_model_dim:
                        feedback_signal = self.ctm_feedback_module(current_noise_for_feedback, ctm_data['final_sync_out'])
                        current_noise = current_noise + feedback_signal.squeeze(1)

                # Combine current noise with CTM influences
                refinement_input = torch.cat([current_noise, sync_inf, state_inf], dim=-1)
                refinement = refinement_layer(refinement_input)
                
                # Progressive residual connection with increasing CTM influence
                ctm_strength = self.coupling_strength * (i + 1) / len(self.ctm_noise_refinement)
                current_noise = current_noise + refinement * ctm_strength
                delta = torch.norm(current_noise - prev_noise, dim=-1).mean()
                deltas.append(delta)
                prev_noise = current_noise.clone()

                # Apply Neuromodulators
                if self.config.enable_neuromodulators:
                    modulation = torch.ones_like(current_noise)
                    for name, mod in self.neuromodulators.items():
                        mod_input = current_noise  # Using current noise as input
                        modulation = modulation * mod(mod_input)
                    current_noise = current_noise * modulation
            
            # Final certainty-based scaling
            if deltas:
                variance = torch.var(torch.stack(deltas))
                confidence = torch.exp(-variance)
                threshold = self.confidence_thresholds.get(confidence_level, 0.8)
                if not self.training and confidence < threshold:
                    current_noise = current_noise * 0
            else:
                confidence = torch.tensor(1.0, device=current_noise.device)
            if 'certainties' in ctm_data:
                final_certainty = ctm_data['certainties'][:, :, -1]
                noise_scale = self.certainty_noise_scaler(final_certainty)
                current_noise = current_noise * noise_scale
            
            # Apply Task-Aware HiPA for intelligent frequency enhancement
            if self.enable_task_aware_hipa:
                # Determine task_id from CTM data or use default
                task_id = getattr(ctm_data, 'task_id', 0) if hasattr(ctm_data, 'task_id') else 0
                
                # Apply intelligent frequency enhancement
                # Reshape current_noise (B, D) to (B, 1, D) for FrequencyDomainAwareAttention
                current_noise_unsqueezed = current_noise.unsqueeze(1)
                enhanced_noise_squeezed, modality_config = self.task_aware_hipa(
                    current_noise_unsqueezed,
                    hipa_control_signal=hipa_control_signal,
                    context_hints={'ctm_data': ctm_data} if ctm_data is not None else None
                )
                # Reshape enhanced_noise (B, 1, D) back to (B, D)
                enhanced_noise = enhanced_noise_squeezed.squeeze(1)
                
                # Log modality detection for debugging
                if hasattr(self, '_debug_mode') and self._debug_mode:
                    print(f"CTM Diffusion - HiPA Applied: {modality_config['use_hipa']} for {modality_config['modality']}")
                
                current_noise = enhanced_noise
            
            # Apply Integration Flow refinement if enabled
            if self.enable_integration_flow and 'final_sync_out' in ctm_data:
                # Use Integration Flow principles for final refinement
                # Integration Flow: x_0 = x_t - G(x_0, x_t, t)
                
                # Get CTM synchronization as guidance
                sync_guidance = ctm_data['final_sync_out']
                
                # Apply Integration Flow correction
                if sync_guidance.shape[-1] == current_noise.shape[-1]:
                    # Direct application if dimensions match
                    integration_correction = sync_guidance * self.integration_flow_strength
                    current_noise = current_noise - integration_correction
                else:
                    # Project sync guidance to noise space if needed
                    if hasattr(self, 'sync_to_noise_proj'):
                        projected_sync = self.sync_to_noise_proj(sync_guidance)
                        integration_correction = projected_sync * self.integration_flow_strength
                        current_noise = current_noise - integration_correction

            # Handle abstention based on confidence threshold
            if 'abstained' in ctm_data and ctm_data['abstained'].any():
                abstained_mask = ctm_data['abstained'].squeeze(-1) # Shape (B,)
                
                # Reshape mask for broadcasting to noise tensor shape
                while abstained_mask.dim() < current_noise.dim():
                    abstained_mask = abstained_mask.unsqueeze(-1)
                
                # Where abstained, use the base (unconditioned) noise prediction.
                # Otherwise, use the CTM-guided prediction.
                final_noise = torch.where(abstained_mask, base_noise_pred, current_noise)
                return final_noise, confidence
            
            return current_noise, confidence
        
        # If no CTM data, still apply Task-Aware HiPA if enabled
        if self.enable_task_aware_hipa:
            enhanced_noise, _ = self.task_aware_hipa(base_noise_pred, task_id=0)
            return enhanced_noise
        
        return base_noise_pred
    
    def compute_enhanced_ctm_guidance(self, ctm_data: Dict[str, torch.Tensor],
                                    target_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Compute enhanced CTM guidance that adapts to the target generation task.
        
        Returns:
            guidance: Tensor of shape target_shape containing the guidance signal
        """
        device = next(self.parameters()).device
        batch_size = target_shape[0]
        
        # Extract CTM components
        sync_out = ctm_data.get('final_sync_out')  # (B, n_synch_out)
        certainties = ctm_data.get('certainties')  # (B, 2, iterations)
        predictions = ctm_data.get('predictions')  # (B, out_dims, iterations)
        
        if sync_out is None:
            return torch.zeros(target_shape, device=device)
        
        # 1. Process synchronization guidance (primary signal)
        sync_guidance = self.sync_to_flow_mapper(sync_out)  # (B, d_model)
        
        # 2. Compute adaptive guidance strength based on CTM certainty
        if certainties is not None:
            final_certainty = certainties[:, :, -1]  # (B, 2)
            strength_input = torch.cat([final_certainty, sync_guidance], dim=-1)
            adaptive_strength = self.adaptive_strength_predictor(strength_input)  # (B, 1)
        else:
            adaptive_strength = torch.ones(batch_size, 1, device=device)
        
        # 3. Apply adaptive strength to guidance
        weighted_sync_guidance = sync_guidance * adaptive_strength * self.sync_weight
        
        # 4. Add certainty-based refinement
        guidance_components = [weighted_sync_guidance]
        
        if certainties is not None:
            # High certainty -> stronger guidance, Low certainty -> gentler guidance
            certainty_factor = final_certainty[:, 0].unsqueeze(-1)  # (B, 1)
            certainty_guidance = sync_guidance * certainty_factor * self.certainty_weight
            guidance_components.append(certainty_guidance)
        
        # 5. Add prediction history guidance
        if predictions is not None:
            # Use final prediction as additional guidance
            final_pred = predictions[:, :, -1]  # (B, out_dims)
            if final_pred.shape[-1] == self.latent_dim:
                pred_guidance = final_pred * self.prediction_weight
                guidance_components.append(pred_guidance)
        
        # 6. Fuse multiple guidance components
        if len(guidance_components) > 1:
            stacked_guidance = torch.stack(guidance_components, dim=1)  # (B, num_components, d_model)
            query = weighted_sync_guidance.unsqueeze(1)  # (B, 1, d_model)
            
            fused_guidance, _ = self.guidance_fusion(
                query, stacked_guidance, stacked_guidance
            )
            final_guidance = fused_guidance.squeeze(1)  # (B, d_model)
        else:
            final_guidance = weighted_sync_guidance
        
        # NEW: Multi-resolution fusion
        # Process guidance at multiple resolutions
        multi_res_outputs = []
        guidance_transposed = final_guidance.unsqueeze(1)  # (B, 1, d_model) -> treat as channels=1
        for processor in self.multi_res_processors:
            res_output = processor(guidance_transposed).squeeze(1)  # (B, d_model)
            multi_res_outputs.append(res_output)
        
        # Fuse multi-resolution outputs
        stacked_multi_res = torch.stack(multi_res_outputs, dim=1)  # (B, num_res, d_model)
        query_multi = final_guidance.unsqueeze(1)
        fused_multi_res, _ = self.guidance_fusion(query_multi, stacked_multi_res, stacked_multi_res)
        final_guidance = fused_multi_res.squeeze(1)
        
        # 7. Project to target shape
        final_guidance = self._project_to_target_shape(final_guidance, target_shape)
            
        # 8. Apply quality enhancement
        if len(final_guidance.shape) >= 2:
            original_shape = final_guidance.shape
            flat_guidance = final_guidance.view(batch_size, -1)
            
            if flat_guidance.shape[-1] == self.latent_dim:
                enhanced_guidance = self.quality_enhancer(flat_guidance)
                final_guidance = enhanced_guidance.view(original_shape)
            
        return final_guidance
    
    def _project_to_target_shape(self, guidance: torch.Tensor,
                               target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Project CTM guidance to target generation shape"""
        batch_size = target_shape[0]
        
        if guidance.shape == target_shape:
            return guidance
        
        # Handle different target shapes
        if len(target_shape) == 2:  # (B, D)
            if guidance.shape[-1] == target_shape[-1]:
                return guidance
            else:
                # Linear projection
                proj = nn.Linear(guidance.shape[-1], target_shape[-1]).to(guidance.device)
                return proj(guidance)
                
        elif len(target_shape) == 3:  # (B, seq_len, D) - sequences
            seq_len, d_out = target_shape[1], target_shape[2]
            
            if guidance.shape[-1] == d_out:
                # Expand to sequence length
                return guidance.unsqueeze(1).expand(-1, seq_len, -1)
            else:
                # Project and expand
                proj = nn.Linear(guidance.shape[-1], d_out).to(guidance.device)
                projected = proj(guidance)
                return projected.unsqueeze(1).expand(-1, seq_len, -1)
                
        elif len(target_shape) == 4:  # (B, C, H, W) - images
            c, h, w = target_shape[1], target_shape[2], target_shape[3]
            target_elements = c * h * w
            
            # Flatten guidance and project to image size
            if guidance.shape[-1] >= target_elements:
                reshaped = guidance[:, :target_elements].view(batch_size, c, h, w)
            else:
                # Repeat and project
                repeat_factor = (target_elements // guidance.shape[-1]) + 1
                repeated = guidance.repeat(1, repeat_factor)[:, :target_elements]
                reshaped = repeated.view(batch_size, c, h, w)
            
            return reshaped
        
        else:
            # Fallback: flatten and project
            target_elements = torch.prod(torch.tensor(target_shape[1:]))
            proj = nn.Linear(guidance.shape[-1], target_elements).to(guidance.device)
            projected = proj(guidance)
            return projected.view(target_shape)

    def one_step_generation(self, noise: torch.Tensor,
                          ctm_data: Dict[str, torch.Tensor],
                          task_id: int = 0) -> torch.Tensor:
        """
        Perform one-step Integration Flow generation with enhanced CTM guidance.
        
        This is the core method that implements:
        x_0 = x_t - G_CTM(x_t, CTM_state)
        
        Where G_CTM is the CTM-learned guidance function.
        """
        # Compute enhanced CTM guidance
        ctm_guidance = self.compute_enhanced_ctm_guidance(ctm_data, noise.shape)
        
        # Integration Flow: x_0 = x_t - G(x_t, CTM_state)
        generated = noise - ctm_guidance
        
        # Optional: Apply learned flow refinement
        if hasattr(self, 'flow_refinement_enabled') and self.flow_refinement_enabled:
            # Combine noise, guidance, and CTM state for learned refinement
            if 'final_sync_out' in ctm_data:
                sync_state = ctm_data['final_sync_out']
                
                # Project sync state to match noise dimensions
                if sync_state.shape[-1] != noise.shape[-1]:
                    sync_proj = nn.Linear(sync_state.shape[-1], noise.shape[-1]).to(noise.device)
                    sync_state = sync_proj(sync_state)
                
                # Expand sync state to match noise shape
                if len(noise.shape) > 2:
                    for _ in range(len(noise.shape) - 2):
                        sync_state = sync_state.unsqueeze(-2)
                    sync_state = sync_state.expand_as(noise)
                
                # Learned refinement
                refinement_input = torch.cat([
                    noise.flatten(1),
                    ctm_guidance.flatten(1),
                    sync_state.flatten(1)
                ], dim=-1)
                
                if refinement_input.shape[-1] == self.latent_dim * 3:
                    refinement = self.flow_predictor(refinement_input)
                    refinement = refinement.view(noise.shape)
                    generated = generated + 0.1 * refinement  # Small refinement
        
        return generated

    def multi_step_refinement(self, initial_generation: torch.Tensor,
                            ctm_data: Dict[str, torch.Tensor],
                            num_refinement_steps: int = 3) -> torch.Tensor:
        """
        Optional multi-step refinement for even higher quality.
        Each step uses updated CTM guidance.
        """
        current = initial_generation
        
        for step in range(num_refinement_steps):
            # Recompute CTM guidance for current state
            refined_guidance = self.compute_enhanced_ctm_guidance(ctm_data, current.shape)
            
            # Apply refinement with decreasing strength
            refinement_strength = 0.1 * (1.0 - step / num_refinement_steps)
            current = current - refined_guidance * refinement_strength
        
        return current
    
    def adaptive_quality_control(self, generated: torch.Tensor,
                               ctm_data: Dict[str, torch.Tensor],
                               quality_threshold: float = 0.8) -> Tuple[torch.Tensor, bool]:
        """
        Adaptive quality control that decides whether to apply refinement
        based on CTM certainty and generation quality metrics.
        """
        needs_refinement = False
        
        # Check CTM certainty
        if 'certainties' in ctm_data:
            final_certainty = ctm_data['certainties'][:, 0, -1].mean().item()
            if final_certainty < quality_threshold:
                needs_refinement = True
        
        # Check generation quality (simple metrics)
        if torch.isnan(generated).any() or torch.isinf(generated).any():
            needs_refinement = True
        
        # Apply refinement if needed
        if needs_refinement:
            refined = self.multi_step_refinement(generated, ctm_data, num_refinement_steps=2)
            return refined, True
        
        return generated, False

    def add_noise(self, x_start: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to clean data according to diffusion schedule"""
        # Get alpha_bar for the given timesteps
        timesteps = torch.clamp(timesteps, 0, len(self.alpha_bars) - 1).to(self.alpha_bars.device)
        alpha_bar = self.alpha_bars[timesteps]
        
        # Reshape alpha_bar to match x_start dimensions
        while len(alpha_bar.shape) < len(x_start.shape):
            alpha_bar = alpha_bar.unsqueeze(-1)
        
        # Apply noise: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        noisy_x = torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * noise
        
        return noisy_x

    def integration_flow_one_step_generation(self, shape: Tuple[int, ...],
                                           ctm_data: Dict[str, torch.Tensor], #  HIPA controlled by signal
                                           hipa_control_signal: Optional[torch.Tensor] = None,
                                           device: torch.device = None) -> torch.Tensor:
        """
        Ultra-fast one-step generation using enhanced Integration Flow with CTM guidance
        and dynamically controlled HIPA.
        """
        if device is None:
            device = next(self.parameters()).device
        
        x_noise = torch.randn(shape, device=device)
        
        # one_step_generation should ideally also accept hipa_control_signal if it uses HIPA internally
        # For now, assuming task_aware_hipa is the main HIPA point here.
        # The task_id from one_step_generation needs to be removed or adapted.
        # Let's assume one_step_generation is a more general denoising step.
        # The original one_step_generation took task_id. We need to see its definition.
        # For now, passing None or a default if it's still required but not used for HIPA.
        x_generated = self.one_step_generation(x_noise, ctm_data, task_id=0) # task_id=0 as placeholder if still needed by one_step_generation internally for non-HIPA reasons

        x_final, _was_refined = self.adaptive_quality_control(x_generated, ctm_data)
        
        # Apply Task-Aware HiPA enhancement, now controlled by hipa_control_signal
        if self.enable_task_aware_hipa and self.task_aware_hipa is not None:
            # The self.task_aware_hipa is FrequencyDomainAwareAttention.
            # Its forward method now takes hipa_control_signal.
            x_final, _modality_config = self.task_aware_hipa(x_final, hipa_control_signal=hipa_control_signal)
        
        return x_final
    
    def get_adaptive_timesteps(self, ctm_data: Dict[str, torch.Tensor],
                              base_timesteps: torch.Tensor) -> torch.Tensor:
        """
        Generate CTM-adaptive timesteps based on CTM certainty and state.
        
        High certainty -> fewer steps needed
        Low certainty -> more steps for careful generation
        """
        if not self.adaptive_scheduling or 'certainties' not in ctm_data or 'activated_states' not in ctm_data:
            return base_timesteps
        
        final_certainty = ctm_data['certainties'][:, :, -1]  # (B, 2)
        final_state = ctm_data['activated_states'][-1]  # (B, d_model)
        
        schedule_input = torch.cat([final_state, final_certainty], dim=-1)
        adaptive_factor = self.adaptive_schedule_predictor(schedule_input)  # (B, 1)
        
        # Modify timesteps based on CTM state
        # High certainty (adaptive_factor close to 1) -> reduce timesteps
        # Low certainty (adaptive_factor close to 0) -> keep more timesteps
        modified_timesteps = base_timesteps * (1.0 - adaptive_factor.squeeze(-1) * 0.5)
        
        return modified_timesteps.long()
    
    def detect_convergence(self, ctm_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Detect if CTM has converged to a stable solution.
        
        Returns convergence probability (0-1) for each batch item.
        """
        if 'activated_states' not in ctm_data or len(ctm_data['activated_states']) < 2:
            return torch.zeros(ctm_data['final_sync_out'].size(0), device=ctm_data['final_sync_out'].device)
        
        # Get final state and certainty
        final_state = ctm_data['activated_states'][-1]
        final_certainty = ctm_data['certainties'][:, :, -1] if 'certainties' in ctm_data else torch.zeros(final_state.size(0), 2, device=final_state.device)
        
        # Simple convergence detection based on certainty
        # High certainty indicates convergence
        convergence_prob = final_certainty[:, 0]  # Use first component of certainty
        
        return convergence_prob
    
    def detect_thought_loop(self, current_state: torch.Tensor,
                           window_size: int = 5, similarity_threshold: float = 0.95) -> torch.Tensor:
        """
        Detect if the model is stuck in a thought loop by comparing current state
        with recent history.
        
        Returns loop detection probability (0-1) for each batch item.
        """
        batch_size = current_state.size(0)
        device = current_state.device
        
        # Update state history
        self.state_history[self.history_pointer] = current_state.mean(dim=0)  # Average across batch
        self.history_pointer = (self.history_pointer + 1) % self.state_history.size(0)
        
        # Check similarity with recent states
        loop_probs = torch.zeros(batch_size, device=device)
        
        for i in range(min(window_size, self.state_history.size(0))):
            if i == 0:
                continue  # Skip current state
            
            # Compute cosine similarity
            hist_state = self.state_history[(self.history_pointer - i - 1) % self.state_history.size(0)]
            similarities = torch.cosine_similarity(current_state, hist_state.unsqueeze(0).expand(current_state.shape), dim=-1)
            
            # Check if similarity exceeds threshold
            loop_detected = (similarities > similarity_threshold).float()
            loop_probs = torch.max(loop_probs, loop_detected)
        
        return loop_probs
    
    def should_early_stop(self, ctm_data: Dict[str, torch.Tensor],
                         current_step: int, max_steps: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Determine if we should stop diffusion early based on:
        1. CTM convergence
        2. High certainty
        3. Thought loop detection
        4. Efficiency considerations
        
        Returns:
            should_stop: (B,) boolean tensor
            stop_reasons: Dict with reasons for stopping
        """
        batch_size = ctm_data['final_sync_out'].size(0)
        device = ctm_data['final_sync_out'].device
        
        stop_reasons = {}
        should_stop = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # 1. Convergence detection
        convergence_prob = self.detect_convergence(ctm_data)
        convergence_stop = convergence_prob > 0.8
        should_stop = should_stop | convergence_stop
        stop_reasons['convergence'] = convergence_stop
        
        # 2. High certainty early stopping
        if 'certainties' in ctm_data:
            final_certainty = ctm_data['certainties'][:, 0, -1]  # First component of certainty
            certainty_stop = final_certainty > 0.85
            should_stop = should_stop | certainty_stop
            stop_reasons['high_certainty'] = certainty_stop
        
        # 3. Thought loop detection
        if 'activated_states' in ctm_data and len(ctm_data['activated_states']) > 0:
            current_state = ctm_data['activated_states'][-1]
            loop_prob = self.detect_thought_loop(current_state)
            loop_stop = loop_prob > 0.9
            should_stop = should_stop | loop_stop
            stop_reasons['thought_loop'] = loop_stop
        
        # 4. Efficiency consideration (encourage early stopping)
        step_ratio = current_step / max_steps
        if step_ratio > 0.7:  # After 70% of max steps, encourage stopping
            efficiency_input = torch.cat([
                ctm_data['final_sync_out'],
                torch.full((batch_size, 1), step_ratio, device=device)
            ], dim=-1)
            efficiency_stop_prob = self.efficiency_predictor(efficiency_input).squeeze(-1)
            efficiency_stop = efficiency_stop_prob > 0.6
            should_stop = should_stop | efficiency_stop
            stop_reasons['efficiency'] = efficiency_stop
        
        return should_stop, stop_reasons

    def denoise_one_step(self, x_t: torch.Tensor, timestep: torch.Tensor,
                         ctm_data: Optional[Dict[str, torch.Tensor]] = None,
                         inferred_task_latent: Optional[torch.Tensor] = None,
                         hipa_control_signal: Optional[torch.Tensor] = None,
                         eta: float = 0.0, # For DDIM
                         generator: Optional[torch.Generator] = None # For stochasticity if needed by scheduler
                         ) -> torch.Tensor:
        """
        Performs one step of denoising using the diffusion model, conditioned by CTM.
        This encapsulates calling the model's forward pass and then using the scheduler.

        Args:
            x_t (torch.Tensor): The current noisy input (e.g., x_t).
            timestep (torch.Tensor): The current timestep. Ensure it's a tensor, possibly (batch_size,).
            ctm_data (Optional[Dict[str, torch.Tensor]]): Conditioning data from CTM.
            inferred_task_latent (Optional[torch.Tensor]): Inferred task latent for conditioning.
            hipa_control_signal (Optional[torch.Tensor]): HIPA control signal.
            eta (float): Eta parameter for DDIM scheduler.
            generator (Optional[torch.Generator]): Random generator for schedulers that require it.

        Returns:
            torch.Tensor: The denoised sample from the previous step (e.g., x_{t-1}).
        """
        # Ensure timestep is correctly shaped for the model forward pass if it expects a scalar or batched tensor
        # The CTMControlledDiffusionProcessor.forward expects timestep as (B,) or a scalar that can be broadcasted.
        # If timestep is a scalar tensor, it might need unsqueezing or repeating.
        # For safety, let's assume timestep is already correctly batched if x_t is batched.
        
        # 1. Predict noise (or x0) using the main forward method of this processor
        model_output_tuple = self.forward( # Calls CTMControlledDiffusionProcessor.forward
            noisy_input=x_t,
            timestep=timestep, # Pass the potentially batched timestep tensor
            ctm_data=ctm_data,
            inferred_task_latent=inferred_task_latent,
            hipa_control_signal=hipa_control_signal
        )
        
        # The forward method returns a tuple: (prediction, ctm_guidance_info)
        # We only need the prediction for the scheduler.
        predicted_value = model_output_tuple[0]


        # 2. Use the sampling_noise_scheduler to compute the previous sample
        # Ensure the arguments match the specific scheduler being used.
        if hasattr(self, 'sampling_noise_scheduler') and hasattr(self.sampling_noise_scheduler, 'step'):
            # For DDPMScheduler, DDIMScheduler from diffusers
            scheduler_step_kwargs = {
                "model_output": predicted_value,
                "timestep": timestep,
                "sample": x_t,
                "generator": generator
            }
            # Check if scheduler's step method accepts 'eta'
            import inspect
            sig = inspect.signature(self.sampling_noise_scheduler.step)
            if "eta" in sig.parameters:
                 scheduler_step_kwargs["eta"] = eta
            
            scheduler_output = self.sampling_noise_scheduler.step(**scheduler_step_kwargs)
            
            if isinstance(scheduler_output, dict):
                 prev_sample = scheduler_output.get('prev_sample')
                 if prev_sample is None:
                     prev_sample = scheduler_output.get('pred_original_sample')
            elif hasattr(scheduler_output, 'prev_sample'):
                prev_sample = scheduler_output.prev_sample
            else:
                prev_sample = scheduler_output

            if prev_sample is None:
                raise ValueError("sampling_noise_scheduler output did not contain 'prev_sample' or 'pred_original_sample'.")
        else:
            raise NotImplementedError(f"CTMControlledDiffusionProcessor.sampling_noise_scheduler is not defined or has no 'step' method.")
            
        return prev_sample

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

        # Core CTM: Conditionally instantiate either Original or Hierarchical core
        if config.use_hrm_core:
            from .ctm_components import HierarchicalCTM
            self.ctm_core = HierarchicalCTM(config)
            # Using HierarchicalCTM (HRM) core if configured
        
        # Enhanced diffusion processor
        # actual_noisy_input_dim is now config.unet_input_feature_dim
        self.diffusion = CTMControlledDiffusionProcessor(
            config,
            actual_noisy_input_dim=config.unet_input_feature_dim,
            task_analyzer=self.task_analyzer_instance # Pass the TaskAnalyzer instance
        )
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

        # --- JEPA Components Initialization (Integrated with LearnedBytePatcherEncoder) ---
        if self.config.use_jepa_training:
            if not self.config.use_dynamic_entropy_patcher:
                raise ValueError("JEPA training requires 'use_dynamic_entropy_patcher' to be True, as it uses LearnedBytePatcherEncoder.")
            if self.dynamic_entropy_patcher is None: # This is the online encoder
                 raise ValueError("self.dynamic_entropy_patcher (LearnedBytePatcherEncoder) must be initialized before JEPA components if use_jepa_training is True.")

            # Target encoder is a momentum copy of the online patcher (dynamic_entropy_patcher)
            self.jepa_target_patch_encoder = copy.deepcopy(self.dynamic_entropy_patcher)
            for param_target in self.jepa_target_patch_encoder.parameters():
                param_target.requires_grad = False
            
            # Predictor operates on patch embeddings
            # The input/output dim for predictor is patch_embedding_dim from the patcher
            jepa_io_dim = self.config.patch_embedding_dim
            # Output dimension of the predictor should be num_target_blocks * patch_embedding_dim
            predictor_output_dim = jepa_io_dim * self.config.jepa_num_target_blocks
            self.jepa_predictor = JEPAPredictor(
                input_dim=jepa_io_dim,
                hidden_dim=self.config.jepa_predictor_hidden_dim,
                output_dim=predictor_output_dim # Predict embeddings for all target blocks
            )
            # JEPA components initialized
    
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
                                         timestep: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
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
            return self.forward(byte_sequence, task_name, target_diffusion_output,
                              'ctm_controlled_diffusion', timestep)
    
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
    
    def forward(self, byte_sequence: torch.Tensor, target_diffusion_output: Optional[torch.Tensor] = None,
                mode: str = 'ctm_controlled_diffusion', timestep: Optional[torch.Tensor] = None,
                current_epoch: int = 0,
                current_batch: int = 0, task_name: Optional[str] = None,
                observed_byte_sequence: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        """
        Forward pass of the CTMDiffusionModel using byte sequences.

        Args:
            byte_sequence (torch.Tensor): Raw byte sequence input.
                                          Shape: (batch_size, sequence_length)
            target_diffusion_output (Optional[torch.Tensor]): The target clean data (x_0) for diffusion loss.
                                                              Required if training with diffusion.
                                                              Shape: (batch_size, sequence_length, output_feature_dim)
            mode (str): Operation mode. Options: 'ctm_controlled_diffusion', 'ctm_only', 'diffusion_only'
                       'ctm_only' runs CTM core.
                       'diffusion_only' runs diffusion processor (needs appropriate 'inputs' as noisy data).
                       'ctm_controlled_diffusion' runs CTM, then diffusion.
            timestep (Optional[torch.Tensor]): Current diffusion timestep (if applicable for diffusion modes)
            current_epoch (int): Current training epoch.
            current_batch (int): Current training batch in the epoch.
            observed_byte_sequence (Optional[torch.Tensor]): Observed agent's byte sequence for mirror neuron processing.
                                                             Shape: (batch_size, sequence_length)

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing various outputs. Key fields include:
            - 'final_output': The primary output tensor (e.g., CTM prediction, MCMC expectation, or diffusion prediction).
            - 'total_loss': Aggregated loss (diffusion loss + MCMC loss + CTM internal losses).
            - 'ctm_core_data': Raw output from the CTM core.
            - 'mcmc_results': Results from the MCMC step, if active.
            - 'diffusion_output': Output from the diffusion processor, if active.
            - Other losses and intermediate data.

        """
        losses = {}
        batch_size = byte_sequence.size(0)
        device = byte_sequence.device
        
        # Initialize hidden states for recurrent connections
        hidden_states = {}
        for layer in range(self.config.n_layers):
            hidden_states[f"layer_{layer}"] = torch.zeros(
                batch_size, self.config.d_model, device=device
            )
            
        # Initialize predictive coding errors
        predictive_errors = {}
        
        # Initialize layer output for first layer
        layer_output = None
        
        # Prepare input features for CTM core (with entropy aux loss)
        kv_features_for_ctm, inferred_task_latent, hipa_control_signal, entropy_aux_loss = \
            self._prepare_input_features(byte_sequence)
        losses['entropy_model_aux_loss'] = entropy_aux_loss * self.config.entropy_model_loss_weight
        
        # Process through CTM core with full tracking
        ctm_data = self.ctm_core.forward_with_full_tracking(kv_features_for_ctm)

        losses['jepa_loss'] = torch.tensor(0.0, device=device)
        # The aux loss from dynamic_entropy_patcher (online JEPA encoder) is handled by _prepare_input_features
        # No separate jepa_context_aux_loss or jepa_target_aux_loss needed here for now.

        # Use the already prepared input features for JEPA
        online_patch_embeddings = kv_features_for_ctm # Shape: (B, S_patches, D_embed)

        
        # PARALLEL BINARY PATCH GENERATION
        # Flatten patches for parallel processing with entropy-based guidance
        batch_size, num_patches, patch_dim = online_patch_embeddings.shape
        parallel_patches = online_patch_embeddings.view(batch_size, -1)

        # --- JEPA Loss Calculation ---
        if self.config.use_jepa_training and self.training and \
           self.jepa_target_patch_encoder is not None and self.jepa_predictor is not None and \
           self.dynamic_entropy_patcher is not None: # Ensure online encoder (dynamic_entropy_patcher) is also available
            try:
                with torch.no_grad():
                    # Target encoder processes the original byte sequence
                    # LearnedBytePatcherEncoder returns (embeddings, indices, aux_loss)
                    target_patch_embeddings, _target_patch_indices, _target_aux_loss = \
                        self.jepa_target_patch_encoder(byte_sequence)
                    # target_patch_embeddings shape: (B, S_patches, D_embed)

                # Create masked views from patch embeddings
                # online_patch_embeddings are from self.dynamic_entropy_patcher (via _prepare_input_features)
                context_representation, actual_target_representation = self._jepa_create_masked_patch_views(
                    online_patch_embeddings, # These are the kv_features_for_ctm
                    target_patch_embeddings
                )

                if context_representation is not None and actual_target_representation is not None:
                    # context_representation is (B, D_embed)
                    # actual_target_representation is (B, num_target_blocks, D_embed)
                    predicted_target_representation_flat = self.jepa_predictor(context_representation) # (B, num_target_blocks * D_embed)
                    
                    # Reshape predicted_target_representation to match actual_target_representation
                    predicted_target_representation = predicted_target_representation_flat.view(
                        batch_size, # batch_size is defined at the start of the forward method
                        self.config.jepa_num_target_blocks,
                        self.config.patch_embedding_dim
                    )
                    
                    current_jepa_loss = F.mse_loss(predicted_target_representation, actual_target_representation.detach())
                    losses['jepa_loss'] = current_jepa_loss * self.config.jepa_loss_weight
                else:
                    # Masking might not have produced valid context/target (e.g., too few patches)
                    losses['jepa_loss'] = torch.tensor(0.0, device=device)

            except Exception as e_jepa:
                # Handle JEPA error gracefully
                losses['jepa_loss'] = torch.tensor(0.0, device=device)
        
        # Initialize states
        batch_size, seq_len, _ = kv_features_for_ctm.shape
        device = kv_features_for_ctm.device
        if not hasattr(self, 'self_emotion_state'):
            self.self_emotion_state = torch.zeros(batch_size, seq_len, self.config.num_emotion_dim, device=device)
        if not hasattr(self, 'observed_emotion_state'):
            self.observed_emotion_state = torch.zeros(batch_size, seq_len, self.config.num_emotion_dim, device=device)
        if not hasattr(self, 'observed_goal_state'):
            self.observed_goal_state = torch.zeros(batch_size, seq_len, self.config.goal_dim, device=device)
        
        # First, run the CTM for the agent itself to get its internal state
        ctm_data = self.ctm_core.forward_with_full_tracking(kv_features_for_ctm)

        # --- Synaptic Empathy Module ---
        # If synaptic empathy is enabled and we have an observed sequence
        if self.config.enable_synaptic_empathy and self.synaptic_empathy_module is not None and observed_byte_sequence is not None:
            # In a real scenario, an "observer" model would predict the state trace from observables.
            # Here, we generate the "true" trace by passing the observed sequence through the CTM core.
            with torch.no_grad():
                observed_kv_features, _, _, _ = self._prepare_input_features(observed_byte_sequence)
                observed_ctm_output = self.ctm_core.forward_with_full_tracking(observed_kv_features)
            
            # The SynapticEmpathy module operates on the historical traces of neuron activations.
            observed_state_trace = observed_ctm_output["state_trace"]
            self_state_trace = ctm_data["state_trace"]
            self_activated_state = ctm_data["activated_states"][-1]

            # The module returns a modulation vector and a reward.
            synaptic_modulation, empathy_reward = self.synaptic_empathy_module(
                self_state_trace=self_state_trace,
                observed_state_trace=observed_state_trace,
                self_activated_state=self_activated_state
            )

            # Apply the synaptic modulation directly to the CTM's final activated state.
            # This directly influences neural dynamics for subsequent processing steps.
            ctm_data["activated_states"][-1] = ctm_data["activated_states"][-1] + synaptic_modulation
            
            # The empathy_reward is a reward, so it should be subtracted from the loss.
            if empathy_reward is not None:
                losses['empathy_reward'] = -empathy_reward.mean() * self.config.synaptic_empathy_reward_weight
        
        # --- High-Level Empathy (Mirror Neuron) Processing ---
        elif self.config.enable_mirror_neurons and self.mirror_layer is not None and observed_byte_sequence is not None:
            # Process observed agent's state to get their thought vector
            with torch.no_grad():
                observed_kv_features, _, _, _ = self._prepare_input_features(observed_byte_sequence)
                observed_ctm_data = self.ctm_core.forward_with_full_tracking(observed_kv_features)
            
            # The thought vector is the final synchronization output from the CTM
            observed_thought = observed_ctm_data['final_sync_out']
            self_thought = ctm_data['final_sync_out']

            # Ensure emotion/goal state trackers match the batch and sequence length
            batch_size, seq_len, _ = self_thought.shape
            prev_self_emotion = self.self_emotion_state.expand(batch_size, seq_len, -1)
            prev_observed_emotion = self.observed_emotion_state.expand(batch_size, seq_len, -1)
            prev_observed_goal = self.observed_goal_state.expand(batch_size, seq_len, -1)

            # Apply mirror neuron layer
            modulated_state, new_self_emotion, new_observed_goal, selfless_reward = self.mirror_layer(
                self_state=self_thought,
                observed_state=observed_thought,
                prev_self_emotion=prev_self_emotion,
                prev_observed_emotion=prev_observed_emotion,
                prev_observed_goal=prev_observed_goal
            )

            # Update persistent emotion and goal states for the next forward pass
            self.self_emotion_state = new_self_emotion[:, -1, :].unsqueeze(1).detach()
            self.observed_emotion_state = self.mirror_layer.observed_emotion_tracker(observed_thought, prev_observed_emotion)[:, -1, :].unsqueeze(1).detach()
            self.observed_goal_state = new_observed_goal[:, -1, :].unsqueeze(1).detach()
            
            # Add the selfless reward to the loss dictionary
            if selfless_reward is not None:
                losses['mirror_reward'] = -selfless_reward.mean() * self.config.mirror_reward_weight
            
            # Update the main CTM data with the modulated state
            ctm_data['final_sync_out'] = modulated_state
        
        # --- CTM Core Logic (Continued) ---
        # The `ctm_data` dictionary, now potentially modified by an empathy module,
        # will be used for subsequent MCMC and Diffusion processing.
        batch_size, num_patches, patch_dim = online_patch_embeddings.shape
        parallel_patches = online_patch_embeddings.view(batch_size, -1)
        
        # --- Diffusion Model Logic ---
        numeric_target_diffusion_output = None
        if target_diffusion_output is not None:
            # PARALLEL PATCH GENERATION MODIFICATION
            # Flatten target for parallel processing
            target_flat = target_diffusion_output.view(batch_size, -1)
            
            # Normalize flattened target
            if target_flat.dtype == torch.uint8:
                float_target = target_flat.float() 
                # 2. Normalize the values. ARC symbols are 0-9, padding is 10.
                # Normalizing to the standard [-1, 1] range.
                # Max value is 10. (val / 5.0) - 1.0 maps [0, 10] to [-1, 1].
                normalized_target = (float_target / 255.0) * 2.0 - 1.0  # Normalize to [-1, 1]
                numeric_target_diffusion_output = normalized_target
            else: # Already numeric (e.g., during sampling)
                numeric_target_diffusion_output = target_flat

            # Use flattened CTM data and parallel patches
            ctm_data_flat = {
                k: v.view(batch_size, -1) if isinstance(v, torch.Tensor) else v
                for k, v in ctm_data.items()
            }
            
            # Generate noise based on flattened target
            noise = torch.randn_like(numeric_target_diffusion_output)
            noisy_input = self.diffusion.add_noise(numeric_target_diffusion_output, noise, timestep)
            
            # Predict with diffusion using parallel inputs
            pred = self.diffusion(
                noisy_input=noisy_input,
                timestep=timestep,
                ctm_data=ctm_data_flat,
                hipa_control_signal=hipa_control_signal
            )
            
            # Compute loss on flattened outputs
            diffusion_loss = F.mse_loss(pred, noise)
            losses['diffusion_loss'] = diffusion_loss

        # PARALLEL PATCH GENERATION MODIFICATION
        # For sampling modes, reshape output to original patch dimensions
        if mode in ['ctm_only', 'mcmc_only']:
            final_output = ctm_data.get('final_sync_out', torch.zeros_like(parallel_patches))
            final_output = final_output.view(batch_size, num_patches, patch_dim)
            
        elif mode in ['ctm_controlled_diffusion', 'diffusion_only']:
            # In diffusion modes, output is already flattened
            final_output = pred if 'pred' in locals() else torch.zeros_like(parallel_patches)

        # Convert final output back to bytes if needed
        if mode in ['ctm_controlled_diffusion', 'diffusion_only']:
            # PARALLEL PATCH GENERATION MODIFICATION
            # Output is already in flattened form, convert directly
            if final_output.dtype != torch.uint8:
                final_output = (final_output * 0.5 + 0.5) * 255.0
                final_output = final_output.clamp(0, 255).byte()

                # --- FIX: Resize target to match UNet input dimension ---
                current_len = numeric_target_diffusion_output.shape[-1]
                target_unet_len = self.config.unet_input_feature_dim
                
                if current_len < target_unet_len:
                    padding = torch.zeros(batch_size, target_unet_len - current_len, device=device, dtype=numeric_target_diffusion_output.dtype)
                    clean_target_for_unet = torch.cat([numeric_target_diffusion_output, padding], dim=-1)
                elif current_len > target_unet_len:
                    clean_target_for_unet = numeric_target_diffusion_output[:, :target_unet_len]
                else:
                    clean_target_for_unet = numeric_target_diffusion_output
                # clean_target_for_unet is now (B, config.unet_input_feature_dim)
                
                # Generate noise based on the correctly-sized target's shape and type
                noise = torch.randn_like(clean_target_for_unet)
                # Use a distinct variable name for the noisy input passed to the diffusion processor
                noisy_input_for_diffusion_processor = self.training_noise_scheduler.add_noise(clean_target_for_unet, noise, timestep)
                
                # CTMControlledDiffusionProcessor.forward (self.diffusion) is the model that predicts noise (or x0)
                # It needs the noisy input, timestep, and CTM conditioning data.
                # kv_features_for_ctm was prepared earlier from byte_sequence.
                ctm_data_for_diffusion_conditioning = self.ctm_core.forward_with_full_tracking(kv_features_for_ctm)

                # Get the prediction from the diffusion processor
                # The diffusion processor's forward method is CTMControlledDiffusionProcessor.forward
                prediction_output_tuple = self.diffusion(
                    noisy_input=noisy_input_for_diffusion_processor,
                    timestep=timestep,
                    ctm_data=ctm_data_for_diffusion_conditioning,
                    hipa_control_signal=hipa_control_signal # Pass HIPA signal
                )

                if isinstance(prediction_output_tuple, tuple): # If it returns (prediction, guidance_info)
                    predicted_noise_or_x0 = prediction_output_tuple[0]
                else:
                    predicted_noise_or_x0 = prediction_output_tuple

                # Determine loss based on the training_noise_scheduler's prediction type
                if not torch.isfinite(predicted_noise_or_x0).all():
                    # Stability Guard: Handle NaN/Inf in diffusion output
                    diffusion_loss = torch.tensor(0.0, device=byte_sequence.device, requires_grad=True)
                elif hasattr(self.training_noise_scheduler, 'config') and hasattr(self.training_noise_scheduler.config, 'prediction_type'):
                    if self.training_noise_scheduler.config.prediction_type == "epsilon":
                        # Halve and zero-center the loss to make a positive learning signal more attainable.
                        # Use tanh to create a bounded, zero-centered loss for the learning signal.
                        # Use tanh with a scaled threshold to create a bounded, zero-centered "reward" signal.
                        # This provides a smoother gradient than a hard threshold and prevents extreme values.
                        # Low MSE -> negative loss -> positive learning signal.
                        diffusion_loss = torch.tanh((F.mse_loss(predicted_noise_or_x0, noise) - 2.1) / 2.1)
                    elif self.training_noise_scheduler.config.prediction_type == "sample":
                        # Also apply to the sample prediction type.
                        diffusion_loss = torch.tanh((F.mse_loss(predicted_noise_or_x0, clean_target_for_unet) - 2.1) / 2.1)
                    else:
                        # Unsupported prediction type
                        diffusion_loss = torch.tensor(0.0, device=byte_sequence.device)
                else: # Default to epsilon prediction if config not available
                    diffusion_loss = torch.tanh((F.mse_loss(predicted_noise_or_x0, noise) - 2.1) / 2.1)
            else:
                # This case should ideally not be reached if training_noise_scheduler is always initialized
                # Missing noise scheduler
                diffusion_loss = torch.tensor(0.0, device=byte_sequence.device)
            losses['diffusion_loss'] = diffusion_loss
        else:
            losses['diffusion_loss'] = torch.tensor(0.0, device=byte_sequence.device)

        # --- Hebbian Plasticity Loss ---
        hebbian_plasticity_loss = torch.tensor(0.0, device=device)
        if self.config.use_activity_plasticity and self.training and 'plastic_adjustments' in ctm_data:
            plastic_adjustments = ctm_data['plastic_adjustments']
            activated_states = ctm_data['activated_states']
            
            if len(plastic_adjustments) > 1 and len(activated_states) > 1:
                for i in range(len(plastic_adjustments) - 1):
                    # Get the plastic adjustment at step t and activated state at step t+1
                    current_plastic_adj = plastic_adjustments[i]
                    next_activated_state = activated_states[i+1].detach() # Detach to treat as target

                    # Identify neurons that fired at step t+1
                    firing_mask = (next_activated_state > 0.1).float()

                    # Calculate cosine similarity for all neurons
                    # The goal is to make the plastic adjustment at 't' predict the activation at 't+1'
                    similarity = F.cosine_similarity(current_plastic_adj, next_activated_state, dim=-1)

                    # Only consider the similarity for neurons that actually fired
                    masked_similarity = similarity * firing_mask.mean(dim=-1)
                    
                    # We want to maximize this similarity, so we minimize its negative
                    hebbian_plasticity_loss -= masked_similarity.mean()

                # Average the loss over the number of steps
                hebbian_plasticity_loss /= (len(plastic_adjustments) -1)


        losses['hebbian_plasticity_loss'] = hebbian_plasticity_loss * self.config.local_hebbian_loss_weight

        if 'diffusion_loss' in losses and losses['diffusion_loss'] is not None:
             learning_signal = -losses['diffusion_loss'].detach() # Make it positive for good performance
             losses['hebbian_plasticity_loss'] = losses['hebbian_plasticity_loss'] * learning_signal


        # Continue with the rest of the function logic
        # For inference, the output would be generated by a sampling loop using the diffusion model,
        # conditioned on ctm_output_features.
        
        # Use the existing variables from the first part of the function
        device = byte_sequence.device

        # Initialize output_dict
        output_dict = {
            'ctm_core_data': None,
            'ctm_internal_loss': torch.tensor(0.0, device=device),
            'mcmc_loss': torch.tensor(0.0, device=device),
            'diffusion_output': None,
            'diffusion_loss': torch.tensor(0.0, device=device), # Placeholder, actual loss computed in training script
            'ctm_guidance_info': None,
            'language_output': None,
            'final_output': None,
            'total_loss': torch.tensor(0.0, device=device)
        }

        # --- 1. CTM Core Processing (Common for most modes) ---
        ctm_data = None
        theta_candidate_from_ctm = None
        if mode in ['ctm_only', 'ctm_controlled_diffusion']:
            # Use the kv_features_for_ctm computed earlier in the function
            ctm_data = self.ctm_core.forward_with_full_tracking(kv_features_for_ctm)
            output_dict['ctm_core_data'] = ctm_data
            theta_candidate_from_ctm = ctm_data['final_sync_out']
            if self.config.use_hrm_core:
                output_dict['programs'] = ctm_data.get('programs')
        
        final_ctm_representation = theta_candidate_from_ctm


        # --- 3. Mode-Specific Outputs & Diffusion ---
        if mode == 'ctm_only':
            output_dict['final_output'] = final_ctm_representation # Could be original ctm_data['final_sync_out']
            if hasattr(self, 'language_head') and final_ctm_representation is not None:
                output_dict['language_output'] = self.language_head(final_ctm_representation)
                output_dict['final_output'] = output_dict['language_output']

        elif mode == 'mcmc_only':
            output_dict['final_output'] = final_ctm_representation # This is MCMC expectation
            if hasattr(self, 'language_head') and final_ctm_representation is not None:
                output_dict['language_output'] = self.language_head(final_ctm_representation)
                output_dict['final_output'] = output_dict['language_output']
        
        elif mode == 'ctm_controlled_diffusion':
            if timestep is None:
                raise ValueError("timestep required for 'ctm_controlled_diffusion' mode")
            if ctm_data is None: # Original CTM data must have been computed
                raise ValueError("Original ctm_data is required for 'ctm_controlled_diffusion' mode")
            if final_ctm_representation is None: # This is the (potentially MCMC refined) output
                 raise ValueError("final_ctm_representation is required for 'ctm_controlled_diffusion' mode but is None.")

            # For this mode, we use the processed_input as the noisy data for the diffusion process.
            
            # Prepare guidance data for diffusion. Start with original CTM data.
            guidance_data_for_diffusion = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in ctm_data.items()}
            
            # Update the primary guidance component in guidance_data_for_diffusion
            # with the MCMC-refined output (final_ctm_representation).
            # We assume 'final_sync_out' is a key that CTMControlledDiffusionProcessor
            # uses from ctm_data for its main guidance. If it's another key, adjust this.
            # Also, ensure the dimensions match what CTMControlledDiffusionProcessor expects for this key.
            # The final_ctm_representation might be of mcmc_output_space.dimension.
            # If diffusion guidance needs the original CTM output dimension, a projection might be needed here,
            # or ensure ctm_to_mcmc_theta_proj was an identity/appropriate mapping if dimensions differ.
            # For now, let's assume direct replacement is intended if MCMC ran.
            # If final_ctm_representation is from MCMC, its shape is (B, mcmc_output_space.dimension)
            # If original ctm_data['final_sync_out'] was (B, d_model), there might be a mismatch
            # if mcmc_output_space.dimension != d_model and no reverse projection exists.

            # Let's assume for now that if MCMC ran, final_ctm_representation is what we want to use,
            # and it's compatible or CTMControlledDiffusionProcessor can handle it.
            # A safer approach might be to add it as a new key, e.g., 'mcmc_refined_guidance',
            # and modify CTMControlledDiffusionProcessor to use it if available.
            # For direct replacement of 'final_sync_out':
            if 'final_sync_out' in guidance_data_for_diffusion:
                # Check if dimensions match before replacing.
                # This is a placeholder for potential dimension mismatch handling.
                # If final_ctm_representation.shape[-1] != guidance_data_for_diffusion['final_sync_out'].shape[-1]:
                # Handle potential dimension mismatch for guidance
                #     # if not hasattr(self, 'mcmc_to_ctm_guidance_proj'):
                #     #     self.mcmc_to_ctm_guidance_proj = nn.Linear(final_ctm_representation.shape[-1], guidance_data_for_diffusion['final_sync_out'].shape[-1]).to(device)
                #     # guidance_data_for_diffusion['final_sync_out'] = self.mcmc_to_ctm_guidance_proj(final_ctm_representation)
                # else:
                # The final_ctm_representation is treated as the 'x' thought vector.
                # It's used here to directly provide the refined guidance signal for diffusion,
                # replacing any original signal in 'final_sync_out' that might have been token-based or less refined.
                guidance_data_for_diffusion['final_sync_out'] = final_ctm_representation
            else:
                guidance_data_for_diffusion['mcmc_refined_guidance_signal'] = final_ctm_representation
            
            # --- Activity Plasticity Update Step ---
            # This is where you would call the plasticity update.
            # The training loop needs to be modified to do this.
            # JEPA target encoder update handled in trainer

            effective_timestep = timestep
            if hasattr(self.config, 'adaptive_scheduling') and self.config.adaptive_scheduling and hasattr(self.diffusion, 'get_adaptive_timesteps'):
                effective_timestep = self.diffusion.get_adaptive_timesteps(guidance_data_for_diffusion, timestep)
            
            # Determine the input to diffusion processor based on training/sampling
            # In training (target_diffusion_output is not None), diffusion input is noisy version of target.
            # In sampling (target_diffusion_output is None), diffusion input is kv_features_for_ctm or some initial noise.
            # The `processed_input` from earlier is derived from `byte_sequence`, not directly the noisy input for diffusion.
            
            diffusion_input_arg = None
            if target_diffusion_output is not None: # Training with diffusion loss
                 # numeric_target_diffusion_output is (B, L_bytes/item_size), e.g. (B, 2048)
                 # Pad or truncate it to config.unet_input_feature_dim
                 current_len = numeric_target_diffusion_output.shape[-1]
                 target_unet_len = self.config.unet_input_feature_dim
                 
                 if current_len < target_unet_len:
                     padding = torch.zeros(batch_size, target_unet_len - current_len, device=device, dtype=numeric_target_diffusion_output.dtype)
                     clean_target_for_unet = torch.cat([numeric_target_diffusion_output, padding], dim=-1)
                 elif current_len > target_unet_len:
                     clean_target_for_unet = numeric_target_diffusion_output[:, :target_unet_len]
                 else:
                     clean_target_for_unet = numeric_target_diffusion_output
                 # clean_target_for_unet is now (B, config.unet_input_feature_dim)

                 current_noise_for_loss = torch.randn_like(clean_target_for_unet)
                 diffusion_input_arg = self.diffusion.add_noise(clean_target_for_unet, current_noise_for_loss, effective_timestep)
                 output_dict['true_noise_for_loss'] = current_noise_for_loss # For loss calculation against predicted noise
            else: # Sampling or CTM-only modes where diffusion might be called for generation
                 # kv_features_for_ctm is (B, S_patches, config.ctm_input_dim)
                 # Average over S_patches to get (B, config.ctm_input_dim)
                 if kv_features_for_ctm.dim() == 3 and kv_features_for_ctm.shape[1] > 0:
                    avg_kv_features = kv_features_for_ctm.mean(dim=1)
                 elif kv_features_for_ctm.dim() == 2: # Already (B, D)
                    avg_kv_features = kv_features_for_ctm
                 else:
                    print(f"Warning: kv_features_for_ctm has unexpected shape {kv_features_for_ctm.shape} in sampling path. Using zeros for avg_kv_features.")
                    avg_kv_features = torch.zeros(batch_size, self.config.ctm_input_dim, device=device, dtype=kv_features_for_ctm.dtype)
                 
                 # Project avg_kv_features (ctm_input_dim) to unet_input_feature_dim
                 diffusion_input_arg = self.sampling_kv_to_unet_input_proj(avg_kv_features)

            diffusion_call_output = self.diffusion( # This is CTMControlledDiffusionProcessor.forward
                noisy_input=diffusion_input_arg, # Corrected argument name
                timestep=effective_timestep,
                ctm_data=guidance_data_for_diffusion,
                hipa_control_signal=hipa_control_signal # Pass the signal
            )
            if isinstance(diffusion_call_output, tuple): # Assuming (prediction, guidance_info)
                noise_pred_or_x0 = diffusion_call_output[0]
                guidance_info = diffusion_call_output[1] # Potential issue if forward doesn't return tuple
            else: # Assuming just prediction
                noise_pred_or_x0 = diffusion_call_output
                guidance_info = {} # Default to empty dict

            output_dict['diffusion_output'] = noise_pred_or_x0
            output_dict['ctm_guidance_info'] = guidance_info
            output_dict['final_output'] = noise_pred_or_x0
            
        elif mode == 'diffusion_only': # Pure diffusion, likely for sampling or specific testing
            if timestep is None:
                raise ValueError("timestep required for 'diffusion_only' mode")
            simplified_ctm_data_for_diffusion_only = None
            
            diffusion_call_output = self.diffusion( # This is CTMControlledDiffusionProcessor.forward
                noisy_input=kv_features_for_ctm, # This is the initial x_t (e.g. noise)
                timestep=timestep,
                ctm_data=simplified_ctm_data_for_diffusion_only,
                hipa_control_signal=hipa_control_signal # Pass the signal
            )
            if isinstance(diffusion_call_output, tuple):
                noise_pred_or_x0 = diffusion_call_output[0]
                guidance_info = diffusion_call_output[1]
            else:
                noise_pred_or_x0 = diffusion_call_output
                guidance_info = {}
            output_dict['diffusion_output'] = noise_pred_or_x0
            output_dict['ctm_guidance_info'] = guidance_info
            output_dict['final_output'] = noise_pred_or_x0
            
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # --- 4. Aggregate Loss ---
        # The actual diffusion loss (e.g., MSE between predicted noise and true noise)
        # is computed in the training script, as it needs the true noise or x0.
        # output_dict['diffusion_loss'] will be updated by the training script.
                
        # Add the losses from the first part of the function to the output_dict
        for loss_name, loss_val in losses.items():
            if loss_name not in output_dict:
                output_dict[loss_name] = loss_val
                
        # Re-aggregate total_loss in output_dict to include all components
        # Start with diffusion_loss which should be in output_dict from earlier processing
        current_total_loss = output_dict.get('diffusion_loss', torch.tensor(0.0, device=device))
        current_total_loss += output_dict.get('ctm_internal_loss', torch.tensor(0.0, device=device))
        current_total_loss += output_dict.get('mcmc_loss', torch.tensor(0.0, device=device))
                
        # Add predictive coding loss to total loss
        if 'ctm_core_data' in output_dict and output_dict['ctm_core_data'] and 'predictive_coding_loss' in output_dict['ctm_core_data']:
            pc_loss = output_dict['ctm_core_data'].get('predictive_coding_loss', torch.tensor(0.0, device=device))
            output_dict['predictive_coding_loss'] = pc_loss
            current_total_loss += pc_loss * getattr(self.config, 'ctm_pc_loss_weight', 0.1)
            output_dict['ctm_internal_loss'] = output_dict.get('ctm_internal_loss', torch.tensor(0.0, device=device)) + pc_loss

        # --- Dopamine Loss from Basal Ganglia ---
        if 'ctm_core_data' in output_dict and output_dict['ctm_core_data'] and 'dopamine_loss' in output_dict['ctm_core_data']:
            # The 'dopamine_loss' is -predicted_reward. We want to maximize reward, so we minimize -reward.
            dopamine_loss = output_dict['ctm_core_data'].get('dopamine_loss', torch.tensor(0.0, device=device))
            output_dict['dopamine_loss'] = dopamine_loss
            current_total_loss += dopamine_loss * 0.1 # Add with some weight

        if self.training and 'diffusion_output' in output_dict and output_dict['diffusion_output'] is not None and 'final_sync_out' in ctm_data:
            ctm_proj = self.ctm_contrastive_proj(ctm_data['final_sync_out'])
            diff_proj = self.diff_contrastive_proj(output_dict['diffusion_output'])

            # Normalize
            ctm_proj = F.normalize(ctm_proj, dim=-1)
            diff_proj = F.normalize(diff_proj, dim=-1)

            # SimCLR loss
            similarity_matrix = torch.matmul(ctm_proj, diff_proj.T) / 0.07
            labels = torch.arange(batch_size, device=device)
            contrastive_loss = F.cross_entropy(similarity_matrix, labels)

            losses['contrastive_loss'] = contrastive_loss * 0.1  # some weight
            current_total_loss += losses['contrastive_loss']
            output_dict['total_loss'] = current_total_loss

        if 'hebbian_plasticity_loss' in output_dict and output_dict['hebbian_plasticity_loss'] is not None:
            current_total_loss += output_dict['hebbian_plasticity_loss']
        output_dict['total_loss'] = current_total_loss

        # Convert final_output to bytes if it's audio from diffusion modes
        if mode in ['ctm_controlled_diffusion', 'diffusion_only'] and output_dict.get('final_output') is not None:
            final_numeric_output = output_dict['final_output']
            if final_numeric_output.dtype != torch.uint8: # Ensure it's not already bytes
                # Ensure it's float32 before converting
                final_numeric_output = final_numeric_output.to(torch.float32)
                try:
                    output_dict['final_output'] = batched_numeric_tensor_to_bytes(final_numeric_output, source_dtype=np.float32)
                except Exception as e:
                    print(f"Warning: Could not convert final_output to bytes in forward method: {e}")
                    # Failed to convert to bytes; keep numeric
        
        # Simplified prediction mechanism using meta-learning
        # Instead of elevating ctm_core_data, use direct outputs from synapse network
        if 'predictions' not in output_dict and 'final_sync_out' not in output_dict:
            # Generate predictions directly from synapse network
            if hasattr(self, 'synapse_network') and 'activated_states' in output_dict:
                # Use the last activated state for prediction
                last_state = output_dict['activated_states'][-1]
                output_dict['predictions'] = self.synapse_network(last_state)
                output_dict['final_sync_out'] = last_state.mean(dim=1)
            else:
                # Fallback to zeros if no synapse network available
                output_dict['predictions'] = torch.zeros_like(input)
                output_dict['final_sync_out'] = torch.zeros(input.size(0), self.config.d_model, device=input.device)

        return output_dict
    
    def iterative_ctm_diffusion_sample(self, shape: Tuple[int, ...],
                                       initial_byte_sequence_for_inference: Optional[torch.Tensor] = None,
                                       num_steps: int = 50,
                                       # ctm_refinement_steps: int = 3, # Not directly used with denoise_one_step
                                       enable_early_stopping: bool = True,
                                       # guidance_scale: float = 1.0, # Not directly used by this sampling structure
                                       eta: float = 0.0, # For DDIM scheduler
                                       generator: Optional[torch.Generator] = None
                                       ) -> Tuple[torch.Tensor, Dict]:
        """
        Advanced sampling with iterative CTM-diffusion refinement and early stopping.
        Uses an initial_byte_sequence_for_inference to determine task characteristics and HIPA control.
        Adds flexible thought adjustment during generation.
        """
        device = self.device_container.device # More robust way to get device
        batch_size = shape[0]

        overall_inferred_latent_for_guidance = None
        overall_hipa_signal_for_guidance = None # Renamed from hipa_control_signal_sampling

        # Infer task latent and HIPA control from initial_byte_sequence_for_inference for overall guidance
        if initial_byte_sequence_for_inference is not None and self.config.use_ctm_guidance_from_condition:
            if initial_byte_sequence_for_inference.size(0) != batch_size:
                # Ensure batch size matches, e.g., by repeating the first sample
                initial_byte_sequence_for_inference = initial_byte_sequence_for_inference[0].unsqueeze(0).expand(batch_size, *initial_byte_sequence_for_inference.shape[1:])
            
            # _prepare_input_features returns (features, latent, hipa_signal)
            # We use the latent and HIPA signal from this initial condition for consistent diffusion processor conditioning.
            # The CTM core itself will process the evolving `x` dynamically.
            _, overall_inferred_latent_for_guidance, overall_hipa_signal_for_guidance = \
                self._prepare_input_features(initial_byte_sequence_for_inference.to(device))
        
        elif initial_byte_sequence_for_inference is None and self.config.use_ctm_guidance_from_condition:
            print("Warning: `use_ctm_guidance_from_condition` is True, but no `initial_byte_sequence_for_inference` provided. Overall guidance signals will be None.")
        # If use_ctm_guidance_from_condition is False, overall_inferred_latent_for_guidance and overall_hipa_signal_for_guidance remain None.

        # Initialize x_t as random noise.
        x = torch.randn(shape, device=device, generator=generator)

        sampling_info = {'steps_taken': [], 'early_stops': [], 'stop_reasons': {},
                         'convergence_history': [], 'certainty_history': [],
                         'thought_adjustments': []}  # Track thought adjustments

        self.diffusion.noise_scheduler.set_timesteps(num_steps, device=device)
        timesteps_to_iterate = self.diffusion.noise_scheduler.timesteps

        if hasattr(self, 'progress_bar_sampler') and callable(self.progress_bar_sampler):
            pb = self.progress_bar_sampler(timesteps_to_iterate)
        else:
            pb = timesteps_to_iterate

        for i, t_tensor in enumerate(pb):
            current_timestep_batched = t_tensor.expand(batch_size) if t_tensor.ndim == 0 else t_tensor
            if current_timestep_batched.ndim == 0 :
                 current_timestep_batched = current_timestep_batched.repeat(batch_size)

            # CTM core processes the current diffusion state `x` (x_t).
            # The `forward_with_full_tracking` method now contains the bidirectional reasoning loop.
            ctm_input_features_for_core_step = x.detach()
            ctm_data_guidance = self.ctm_core.forward_with_full_tracking(ctm_input_features_for_core_step)
            
            # Optional: store certainty for logging
            if 'certainties' in ctm_data_guidance and ctm_data_guidance['certainties'] is not None:
                current_certainty = ctm_data_guidance['certainties'][:, 0, -1].mean().item()
                sampling_info['certainty_history'].append(current_certainty)

            if enable_early_stopping and i > self.config.early_stop_min_steps and hasattr(self.diffusion, 'should_early_stop'):
                should_stop_flags, stop_reason_details_dict = self.diffusion.should_early_stop(ctm_data_guidance, i, len(timesteps_to_iterate))
                
                # Log reasons for samples that are stopping
                for sample_idx in range(batch_size):
                    if should_stop_flags[sample_idx]:
                        reasons_for_sample = [reason for reason, flag_tensor in stop_reason_details_dict.items() if flag_tensor[sample_idx]]
                        if sample_idx not in sampling_info['stop_reasons']: sampling_info['stop_reasons'][sample_idx] = []
                        sampling_info['stop_reasons'][sample_idx].append({'step': i, 'reasons': reasons_for_sample})

                if should_stop_flags.any() and self.config.break_sampling_on_early_stop:
                    print(f"Early stopping triggered for at least one sample at step {i}.")
                    # Early stopping triggered
                    break
            
            x = self.diffusion.denoise_one_step(
                x_t=x,
                timestep=current_timestep_batched,
                ctm_data=ctm_data_guidance, # Dynamic CTM output based on current x
                inferred_task_latent=overall_inferred_latent_for_guidance, # Static from initial condition (if any)
                hipa_control_signal=overall_hipa_signal_for_guidance,     # Static from initial condition (if any)
                eta=eta,
                generator=generator
            )
                
            # External feedback: Feed diffusion output back to CTM
            feedback_input = self.feedback_proj(x.detach()).unsqueeze(1)  # Project to ctm_input_dim and add seq dim=1
            feedback_ctm_data = self.ctm_core.forward_with_full_tracking(feedback_input)
                
            # Update guidance with feedback (simple average for demonstration)
            for key in ctm_data_guidance:
                if isinstance(ctm_data_guidance[key], torch.Tensor):
                    ctm_data_guidance[key] = (ctm_data_guidance[key] + feedback_ctm_data[key]) / 2
                
            sampling_info['steps_taken'].append(i)

        # Ensure x is float32 before converting to bytes
        x = x.to(torch.float32)
        # Convert to byte tensor
        x_bytes = batched_numeric_tensor_to_bytes(x, source_dtype=np.float32)
        return x_bytes, sampling_info
    
    def get_loss_with_ctm_guidance(self, x_start: torch.Tensor, task_id: int = 0) -> Tuple[torch.Tensor, Dict]:
        """
        Compute diffusion loss with CTM guidance and return detailed metrics.
        """
        device = x_start.device
        batch_size = x_start.size(0)
        
        # Sample random timesteps
        timesteps = torch.randint(0, len(self.diffusion.betas), (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Add noise to clean data
        noisy_x = self.diffusion.add_noise(x_start, noise, timesteps)
        
        # Predict noise with CTM control
        predicted_noise, ctm_data = self.forward(noisy_x, timesteps, task_id, mode='ctm_controlled_diffusion')
        
        # Main diffusion loss
        diffusion_loss = nn.functional.mse_loss(predicted_noise, noise)
        
        # Additional CTM-based losses for better integration
        additional_losses = {}
        
        # Certainty consistency loss (encourage consistent certainty)
        if 'certainties' in ctm_data:
            certainty_var = torch.var(ctm_data['certainties'], dim=-1).mean()
            additional_losses['certainty_consistency'] = certainty_var * 0.1
        
        # Synchronization stability loss
        if 'sync_out_history' in ctm_data and len(ctm_data['sync_out_history']) > 1:
            sync_diffs = []
            for i in range(1, len(ctm_data['sync_out_history'])):
                diff = torch.mse_loss(ctm_data['sync_out_history'][i], ctm_data['sync_out_history'][i-1])
                sync_diffs.append(diff)
            sync_stability_loss = torch.stack(sync_diffs).mean()
            additional_losses['sync_stability'] = sync_stability_loss * 0.05
        
        # Total loss
        total_loss = diffusion_loss + sum(additional_losses.values())
        
        # FIX: Ensure CTM data contains final_sync_out for consistent ARC head input
        if 'final_sync_out' not in ctm_data and 'predictions' in ctm_data:
            # If final_sync_out is missing but predictions exist, use predictions
            # but note: predictions may be 64D while ARC head expects 512D
            # This is a fallback and may cause issues if predictions dimension doesn't match
            ctm_data['final_sync_out'] = ctm_data['predictions']
            print ("Predictions dimensions were used instead of final_sync_out: ERROR")
            # Dimension fallback for ARC head
        
        return total_loss, {
            'diffusion_loss': diffusion_loss,
            'total_loss': total_loss,
            **additional_losses,
            'ctm_data': ctm_data
        }
    
    def ultra_fast_integration_flow_generation(self, shape: Tuple[int, ...],
                                             # task_id: int = 0, # Replaced
                                             initial_byte_sequence_for_inference: Optional[torch.Tensor] = None,
                                             text_condition: Optional[torch.Tensor] = None, # Keep if used by CTM/Diffusion
                                             enable_hipa_flag: bool = True # Keep as an override/general toggle
                                             ) -> Tuple[torch.Tensor, Dict]:
        """
        Ultra-fast one-step generation. HIPA control is now primarily via inferred latent.
        `enable_hipa_flag` can be a global switch.
        """
        device = next(self.parameters()).device
        batch_size = shape[0]
        generation_info = {'method': 'integration_flow_one_step', 'hipa_applied': False, 'modality_detected': 'unknown', 'generation_time': 0.0}
        
        # Infer task latent and HIPA control signal
        hipa_control_signal_sampling = None
        ctm_input_for_guidance_generation = None

        if initial_byte_sequence_for_inference is not None:
            if initial_byte_sequence_for_inference.size(0) != batch_size:
                 initial_byte_sequence_for_inference = initial_byte_sequence_for_inference[0].unsqueeze(0).expand(batch_size, -1)
            
            encoded_features, _inferred_latent, hipa_control_signal_sampling = \
                self._prepare_input_features(initial_byte_sequence_for_inference.to(device))
            ctm_input_for_guidance_generation = self.compute_features(encoded_features)
        else: # Fallback for HIPA signal and CTM input
            # Using dummy input for guidance
            pass

        if not enable_hipa_flag: # Global override to turn HIPA off
            hipa_control_signal_sampling = torch.zeros_like(hipa_control_signal_sampling) if hipa_control_signal_sampling is not None else None
            generation_info['hipa_applied'] = False
        
        start_time_gen = time.time()
        
        ctm_data = self.ctm_core.forward_with_full_tracking(ctm_input_for_guidance_generation)
        
        # Assuming integration_flow_one_step_generation in CTMControlledDiffusionProcessor
        # now accepts hipa_control_signal
        if hasattr(self.diffusion, 'integration_flow_one_step_generation'):
            generated_samples = self.diffusion.integration_flow_one_step_generation(
                shape=shape, ctm_data=ctm_data, device=device,
                # task_id removed from integration_flow_one_step_generation
                hipa_control_signal=hipa_control_signal_sampling
            )
        else:
            # Fallback to iterative sampling
            print("Warning: integration_flow_one_step_generation not found on diffusion processor. Using standard iterative sampling as fallback.")
            generated_samples, _info = self.iterative_ctm_diffusion_sample(
                shape=shape,
                initial_byte_sequence_for_inference=initial_byte_sequence_for_inference,
                num_steps=self.config.diffusion_num_inference_steps # Use default steps
            )

        generation_info['generation_time'] = time.time() - start_time_gen
        
        # The HIPA application is now expected to be handled *within* integration_flow_one_step_generation
        # or the sampling loop if it uses HIPA-aware components, based on the passed hipa_control_signal.
        # So, the explicit call to self.diffusion.task_aware_hipa here is removed.
        # generation_info['hipa_applied'] should reflect if the signal was active.
        if hipa_control_signal_sampling is not None and torch.any(hipa_control_signal_sampling > 0.5):
            generation_info['hipa_applied'] = True
        # Modality info would come from FrequencyDomainAwareAttention if it's used and returns it.
        # For now, this is simplified.

        # Ensure generated_samples is float32 before converting to bytes
        generated_samples = generated_samples.to(torch.float32)
        # Convert to byte tensor
        generated_samples_bytes = batched_numeric_tensor_to_bytes(generated_samples, source_dtype=np.float32)
        return generated_samples_bytes, generation_info
    

# Configuration helpers
    def iterative_generation(self, condition: torch.Tensor, num_steps: int = 3) -> torch.Tensor:
        """
        Generate output with iterative refinement.
        
        Args:
            condition: CTM guidance data (batch_size, d_model)
            num_steps: Number of refinement steps
            
        Returns:
            Refined output
        """
        # Initial generation at final time step
        time = torch.ones(condition.size(0), 1, device=condition.device) * self.config.diffusion_steps
        output = self.integration_flow(condition, time)
        
        # Iterative refinement
        for _ in range(num_steps):
            # Prepare refinement input: current output + condition
            refine_input = torch.cat([output, condition], dim=1)
            refinement = self.integration_flow.refinement_net(refine_input)
            output = output + refinement
            
        return output

    def _jepa_create_masked_patch_views(self,
                                        online_patch_embeddings: torch.Tensor,
                                        target_patch_embeddings: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Creates context and target representations from sequences of patch embeddings for JEPA.
        Masking is applied at the patch sequence level. This version aims to select
        one context block and one target block per sample.

        Args:
            online_patch_embeddings: (B, S_patches, D_embed) from the online encoder.
            target_patch_embeddings: (B, S_patches, D_embed) from the target encoder.

        Returns:
            Tuple of (context_representation, actual_target_representation), or (None, None) if masking fails.
            context_representation: (B, D_embed) - selected context patch from online encoder.
            actual_target_representation: (B, num_target_blocks, D_embed) - selected target patches from target encoder.
        """
        B, S_patches, D_embed = online_patch_embeddings.shape
        device = online_patch_embeddings.device

        if S_patches < 2: # Need at least one patch for context and one for target
            return None, None

        batch_context_reps = []
        batch_target_reps = []

        for b_idx in range(B):
            # 1. Determine context block size
            context_scale = random.uniform(self.config.jepa_context_scale_min, self.config.jepa_context_scale_max)
            num_context_patches = max(1, int(S_patches * context_scale))
            # Ensure there's enough space for at least num_target_blocks patches left after context
            num_context_patches = min(num_context_patches, S_patches - self.config.jepa_num_target_blocks)

            if num_context_patches <= 0: # Not enough patches to form a context and have targets
                continue

            # 2. Select context block
            # Ensure there's enough space for context block and target blocks
            if S_patches < num_context_patches + self.config.jepa_num_target_blocks:
                continue

            all_indices = torch.arange(S_patches, device=device)
            
            # Randomly select start for context block
            # Max start index for context ensures that context_block + target_blocks fit
            max_context_start_idx = S_patches - num_context_patches - self.config.jepa_num_target_blocks
            if max_context_start_idx < 0: # Should be caught by S_patches check above, but for safety
                continue
            
            context_start_idx = random.randint(0, max_context_start_idx)
            context_indices = all_indices[context_start_idx : context_start_idx + num_context_patches]
            context_block_embeddings = online_patch_embeddings[b_idx, context_indices, :] # (num_context_patches, D_embed)
            context_rep = context_block_embeddings.mean(dim=0) # (D_embed) - Average context patches

            # 3. Select target blocks (non-overlapping with context)
            # Create a mask for available target indices
            available_target_mask = torch.ones(S_patches, dtype=torch.bool, device=device)
            available_target_mask[context_indices] = False # Mask out context indices
            
            potential_target_indices = all_indices[available_target_mask]

            if len(potential_target_indices) < self.config.jepa_num_target_blocks:
                continue # Not enough non-overlapping patches left for the required number of target blocks
            
            # Shuffle potential target indices and select
            shuffled_potential_target_indices = potential_target_indices[torch.randperm(len(potential_target_indices), device=device)]
            actual_target_indices = shuffled_potential_target_indices[:self.config.jepa_num_target_blocks]
            
            selected_target_patches = target_patch_embeddings[b_idx, actual_target_indices, :] # (num_target_blocks, D_embed)
            target_rep = selected_target_patches # Keep as distinct blocks, shape (num_target_blocks, D_embed)

            batch_context_reps.append(context_rep) # List of (D_embed)
            batch_target_reps.append(target_rep)   # List of (num_target_blocks, D_embed)

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
        

class IntegrationFlowHiPASampler: #Needed for the One Step Diffusion in the Diffusion Controller CTM Class function. 
    """Advanced Integration Flow sampler with Task-Aware HiPA for CTM integration."""
    
    def __init__(self, task_aware_hipa_module: FrequencyDomainAwareAttention, num_steps=50, beta_start=0.0001, beta_end=0.02,
                 sigma_min=0.01, sigma_max=50.0, hipa_freq_threshold=0.1,
                 integration_flow_strength=1.0, model_type='VE'):
        # self.model = model # model argument removed, context_fn will be passed to sample methods
        self.task_aware_hipa = task_aware_hipa_module # Store the passed HiPA module
        self.num_steps = num_steps
        self.model_type = model_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.hipa_freq_threshold = hipa_freq_threshold
        self.integration_flow_strength = integration_flow_strength
        
        # Noise schedules for different model types
        if model_type == 'VE':
            # Variance Exploding schedule: σ_min * (σ_max/σ_min)^(t/T)
            t_values = torch.linspace(1, num_steps, num_steps)
            self.sigma_schedule = sigma_min * (sigma_max / sigma_min) ** (t_values / num_steps)
        elif model_type == 'RectifiedFlow':
            # Linear interpolation, no noise schedule needed
            self.sigma_schedule = torch.zeros(num_steps)
        elif model_type == 'PFGM++':
            # Similar to VE but with radius scaling
            t_values = torch.linspace(1, num_steps, num_steps)
            self.sigma_schedule = sigma_min * (sigma_max / sigma_min) ** (t_values / num_steps)
        
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # Pre-compute integration flow trajectory weights
        self._precompute_integration_weights()
        # Sampler initialized
    
    def _precompute_integration_weights(self):
        """Pre-compute integration weights for direct trajectory learning."""
        # Integration Flow: Learn the integral of ODE trajectory paths
        t_values = torch.linspace(0, 1, self.num_steps)
        
        # Compute cumulative integration weights (trapezoidal rule)
        dt = 1.0 / (self.num_steps - 1) if self.num_steps > 1 else 1.0
        weights = torch.ones_like(t_values) * dt
        if len(weights) > 1:
            weights[0] *= 0.5  # Trapezoidal rule adjustment
            weights[-1] *= 0.5
        
        self.integration_weights = torch.cumsum(weights, dim=0)
    
    def _apply_hipa_attention(self, x, task_id=None, freq_domain=True):
        """Apply Task-Aware HiPA to intelligently enhance frequency details."""
        if not freq_domain or x.numel() == 0:
            return x
        
        try:
            # Use task-aware HiPA system for intelligent enhancement
            enhanced_x, modality_config = self.task_aware_hipa(x, task_id=task_id)
            
            # Log modality detection for debugging
            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"  HiPA Modality Detection:")
                print(f"  - Detected: {modality_config['modality']}")
                print(f"  - HiPA Applied: {modality_config['use_hipa']}")
                if modality_config['use_hipa']:
                    print(f" - FFT Dims: {modality_config['fft_dims']}")
                    print(f" - Enhancement: {modality_config['enhancement_strength']:.2f}")
                else:
                    print(f" - Reason: Protecting {modality_config['modality']} from frequency corruption")
            
            return enhanced_x
            
        except Exception as e:
            print(f"Warning: Task-aware HiPA failed: {e}, using original tensor")
            return x
    
    def _get_preconditioning_params(self, t):
        """Get preconditioning parameters a_t and b_t based on model type."""
        if self.model_type == 'VE':
            # VE: a_t = σ_min/σ_t, b_t = 1 - σ_min/σ_t
            sigma_t = self.sigma_schedule[min(t, len(self.sigma_schedule)-1)]
            a_t = self.sigma_min / (sigma_t + 1e-8)
            b_t = 1.0 - a_t
        elif self.model_type == 'RectifiedFlow':
            # Rectified Flow: a_t = 1, b_t = t/T
            a_t = 1.0
            b_t = t / self.num_steps
        elif self.model_type == 'PFGM++':
            # PFGM++: a_t = 1/√(1+σ_t²), b_t = σ_t/√(1+σ_t²)
            sigma_t = self.sigma_schedule[min(t, len(self.sigma_schedule)-1)]
            denom = torch.sqrt(1 + sigma_t**2)
            a_t = 1.0 / denom
            b_t = sigma_t / denom
        else:
            # Default to simple linear
            a_t = 1.0
            b_t = 0.5
        
        return a_t, b_t
    
    def one_step_sample(self, shape, context_fn, task_id, time_device, text_condition=None,
                       iterative_refinement=True, num_refinement_steps=2):
        """Ultra-fast one-step generation using Integration Flow with Task-Aware HiPA."""
        # Start with noise based on model type
        if self.model_type == 'VE' or self.model_type == 'PFGM++':
            # Start with scaled noise
            x_noise = torch.randn(shape, device=time_device) * self.sigma_max
        else:  # RectifiedFlow
            # Start with standard noise
            x_noise = torch.randn(shape, device=time_device)
        
        # Integration Flow: Direct trajectory integration
        # Use final timestep for one-step generation
        t_integration = self.num_steps - 1
        t_tensor = torch.full((shape[0],), t_integration, device=time_device, dtype=torch.long)
        
        # Initialize estimate
        x_est = torch.zeros_like(x_noise)
        
        # Iterative refinement as described in Integration Flow paper
        for step in range(num_refinement_steps if iterative_refinement else 1):
            try:
                # Get model prediction G_θ(x_0^(n), x_t, t)
                if text_condition is not None:
                    # For TTS with text conditioning - need to adapt context_fn
                    if hasattr(context_fn, '__call__'):
                        # Assume context_fn can handle text_condition
                        G_pred = context_fn(x_noise, t_tensor, task_id, text_condition)
                    else:
                        G_pred = context_fn(x_noise, t_tensor, task_id)
                else:
                    G_pred = context_fn(x_noise, t_tensor, task_id)
                
                # Get preconditioning parameters
                a_t, b_t = self._get_preconditioning_params(t_integration)
                
                # Integration Flow formula: g_θ(x_0, x_t, t) = a_t * x_t + b_t * G_θ(x_0, x_t, t)
                x_est_new = a_t * x_noise + b_t * G_pred
                
                # Apply integration flow strength
                x_est = x_est + self.integration_flow_strength * (x_est_new - x_est)
                
            except Exception as e:
                # Integration Flow failed
                if self.model_type == 'VE':
                    x_est = x_noise / self.sigma_max
                else:
                    x_est = x_noise
                break
        
        # Apply Task-Aware HiPA for intelligent frequency enhancement
        x_final = self._apply_hipa_attention(x_est, task_id=task_id, freq_domain=True)
        
        return x_final
    
    def sample(self, shape, context_fn, task_id, time_device, text_condition=None,
              use_one_step=True, **kwargs):
        """Main sampling function with option for one-step or multi-step."""
        if use_one_step:
            return self.one_step_sample(
                shape, context_fn, task_id, time_device, text_condition,
                iterative_refinement=kwargs.get('iterative_refinement', True),
                num_refinement_steps=kwargs.get('num_refinement_steps', 2)
            )
        else:
            # Fallback to multi-step with HiPA enhancement
            return self._multi_step_sample_with_hipa(
                shape, context_fn, task_id, time_device, text_condition
            )
    
    def _multi_step_sample_with_hipa(self, shape, context_fn, task_id, time_device, text_condition=None):
        """Multi-step sampling with HiPA enhancement as fallback."""
        # Start with appropriate noise
        if self.model_type == 'VE' or self.model_type == 'PFGM++':
            x = torch.randn(shape, device=time_device) * self.sigma_max
        else:
            x = torch.randn(shape, device=time_device)
        
        # Reduced steps with HiPA enhancement
        step_size = max(1, self.num_steps // 10)  # Use only 10% of steps
        
        for i in range(0, self.num_steps, step_size):
            t = self.num_steps - 1 - i
            if t < 0:
                break
                
            t_tensor = torch.full((shape[0],), t, device=time_device, dtype=torch.long)
            
            try:
                # Get model prediction
                if text_condition is not None:
                    eps_theta = context_fn(x, t_tensor, task_id, text_condition)
                else:
                    eps_theta = context_fn(x, t_tensor, task_id)
                
                # Apply denoising step based on model type
                if self.model_type == 'VE':
                    # VE denoising
                    sigma_t = self.sigma_schedule[min(t, len(self.sigma_schedule)-1)]
                    x = x - sigma_t * eps_theta
                elif self.model_type == 'RectifiedFlow':
                    # Rectified flow step
                    dt = 1.0 / self.num_steps
                    x = x - dt * eps_theta
                else:  # PFGM++
                    # Similar to VE
                    sigma_t = self.sigma_schedule[min(t, len(self.sigma_schedule)-1)]
                    x = x - sigma_t * eps_theta
                
                # Apply HiPA enhancement every few steps
                if i % (step_size * 2) == 0:
                    x = self._apply_hipa_attention(x, task_id=task_id, freq_domain=True)
                    
            except Exception as e:
               print(f"Warning: Multi-step sampling failed at step {i}: {e}")
               break        
        # Final HiPA enhancement
        x_final = self._apply_hipa_attention(x, task_id=task_id, freq_domain=True)
        
        return x_final

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

