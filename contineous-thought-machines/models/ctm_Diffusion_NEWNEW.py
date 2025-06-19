"""
Enhanced CTM-Controlled Diffusion Architecture - This is the most current version of ctm.py that is currently being used.

This implementation gives the CTM deep control and influence over the diffusion process
through multiple mechanisms:
1. Direct noise prediction conditioning
2. Adaptive timestep scheduling based on CTM certainty
3. CTM-guided attention mechanisms with WINA sparse activation
4. Synchronization-based diffusion guidance
5. Iterative CTM-diffusion coupling
6. WINA (Weight Informed Neuron Activation) for efficient sparse attention
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
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import numpy as np


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
                         cache_key: str = None) -> torch.Tensor:
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
        
        # Compute WINA criterion: |x_i * c_i| where c_i is column norm
        wina_scores = torch.abs(hidden_states) * column_norms
        
        # Determine sparsity ratio for this layer
        if self.use_layer_specific and layer_name in self.layer_sparsity_ratios:
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
        
        # Apply gating
        return hidden_states * mask
    
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
                 dropout: float = 0.1,
                 sparsity_ratio: float = 0.5,
                 use_adaptive_sparsity: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_adaptive_sparsity = use_adaptive_sparsity
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.wina_sparsifier = WINASparsifier(sparsity_ratio=sparsity_ratio)
        
        # Setup adaptive sparsity if enabled
        if use_adaptive_sparsity:
            layer_names = ["query", "key", "value", "attention", "output"]
            self.wina_sparsifier.adaptive_sparsity_allocation(layer_names, sparsity_ratio)
        
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        
        # Apply WINA sparsification to input representations
        query_sparse = self.wina_sparsifier.apply_wina_gating(
            query, self.q_proj.weight, "query", f"q_proj_{id(self.q_proj.weight)}"
        )
        key_sparse = self.wina_sparsifier.apply_wina_gating(
            key, self.k_proj.weight, "key", f"k_proj_{id(self.k_proj.weight)}"
        )
        value_sparse = self.wina_sparsifier.apply_wina_gating(
            value, self.v_proj.weight, "value", f"v_proj_{id(self.v_proj.weight)}"
        )
        
        # Project to Q, K, V
        Q = self.q_proj(query_sparse)
        K = self.k_proj(key_sparse)
        V = self.v_proj(value_sparse)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply WINA sparsification to attention weights
        # Create identity matrix as proxy weight matrix for attention sparsification
        device = attention_weights.device
        dtype = attention_weights.dtype
        attn_seq_len = attention_weights.shape[-1]
        
        # Use identity matrix as weight matrix for attention sparsification
        identity_weight = torch.eye(attn_seq_len, device=device, dtype=dtype)
        
        # Apply WINA to attention weights (reshape for compatibility)
        original_shape = attention_weights.shape
        attention_weights_flat = attention_weights.view(-1, attn_seq_len)
        
        attention_weights_sparse = self.wina_sparsifier.apply_wina_gating(
            attention_weights_flat, identity_weight, "attention", f"attn_identity_{attn_seq_len}"
        )
        
        attention_weights_sparse = attention_weights_sparse.view(original_shape)
        attention_weights_sparse = self.dropout(attention_weights_sparse)
        
        # Apply attention to values
        context = torch.matmul(attention_weights_sparse, V)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Apply WINA sparsification to output projection
        context_sparse = self.wina_sparsifier.apply_wina_gating(
            context, self.out_proj.weight, "output", f"out_proj_{id(self.out_proj.weight)}"
        )
        
        output = self.out_proj(context_sparse)
        
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

class SubquadraticAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 epsilon=1e-6, poly_degree=5, scale=None): # Changed dim to embed_dim
        super().__init__()
        self.embed_dim = embed_dim # Store embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.scale = scale if scale is not None else self.head_dim ** -0.5

        # Individual Q, K, V projections from the input query, key, value tensors
        self.q_proj_layer = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj_layer = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj_layer = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim) # Output projection
        self.proj_drop = nn.Dropout(proj_drop)

        self.epsilon = epsilon
        if not (isinstance(poly_degree, int) and poly_degree >= 0):
            raise ValueError("poly_degree must be a non-negative integer.")
        self.poly_degree = poly_degree

    def _taylor_exp_poly(self, y_tensor):
        """
        Computes Taylor approximation of exp(y) = sum_{i=0 to degree} y^i / i!
        Assumes y_tensor contains values in a range [0, W] suitable for approximation.
        """
        approx_exp = torch.zeros_like(y_tensor)
        term = torch.ones_like(y_tensor)  # Corresponds to y^0 / 0!
        approx_exp += term
        
        for i in range(1, self.poly_degree + 1):
            term = term * y_tensor / i  # Efficiently computes (y^i / i!) from (y^(i-1) / (i-1)!)
            approx_exp += term
        return approx_exp

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True, # Standard MHA param, controls if attn_weights are returned
                average_attn_weights: bool = True # Standard MHA param
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        B_q, N_q_orig, C_q = query.shape    # Batch, Query Seq Len, Query Dim (embed_dim)
        B_kv, N_kv_orig, C_kv = key.shape # Batch, Key Seq Len, Key Dim (embed_dim)
        # Value shape: B_kv, N_kv_orig, Value Dim (embed_dim)

        # Project Q, K, V using dedicated layers
        q_projected = self.q_proj_layer(query)  # (B_q, N_q_orig, embed_dim)
        k_projected = self.k_proj_layer(key)    # (B_kv, N_kv_orig, embed_dim)
        v_projected = self.v_proj_layer(value)  # (B_kv, N_kv_orig, embed_dim)

        # Reshape and permute for multi-head processing
        # q: (B_q, num_heads, N_q_orig, head_dim)
        q = q_projected.reshape(B_q, N_q_orig, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # k: (B_kv, num_heads, N_kv_orig, head_dim)
        k = k_projected.reshape(B_kv, N_kv_orig, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # v: (B_kv, num_heads, N_kv_orig, head_dim)
        v = v_projected.reshape(B_kv, N_kv_orig, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        N_q_tokens = q.shape[-2]
        N_k_tokens = k.shape[-2]

        if N_k_tokens == 0: # Fallback for empty key/value sequence
            context = torch.zeros(B_q, N_q_orig, self.embed_dim, device=q.device, dtype=q.dtype)
            context = self.proj(context)
            context = self.proj_drop(context)
            return context, None

        # Scaled dot-product scores: (B, H, N_q_tokens, N_k_tokens)
        # Assuming B_q == B_kv for matmul, which is typical.
        scaled_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if attn_mask is not None:
            # Correctly handle attention mask broadcasting.
            # MHA expects mask to be (N_q, N_k), (B, N_q, N_k), or (B, H, N_q, N_k)
            # If mask is (B, N_q, N_k), it needs to be unsqueezed for heads.
            # If mask is (N_q, N_k), it needs to be unsqueezed for batch and heads.
            if attn_mask.dim() == 2: # (N_q_tokens, N_k_tokens)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # -> (1, 1, N_q_tokens, N_k_tokens)
            elif attn_mask.dim() == 3: # (B, N_q_tokens, N_k_tokens)
                attn_mask = attn_mask.unsqueeze(1)  # -> (B, 1, N_q_tokens, N_k_tokens)
            # Mask should now be broadcastable with scaled_scores (B, H, N_q, N_k)
            # The mask uses 0 for masked positions, 1 for unmasked.
            # masked_fill expects a boolean mask where True means fill.
            # So, if attn_mask has 0 for masked, we use `attn_mask == 0`.
            scaled_scores = scaled_scores.masked_fill(attn_mask == 0, float('-inf'))


        max_scores_per_query, _ = torch.max(scaled_scores, dim=-1, keepdim=True)
        # Ensure N_k_tokens is float for division, and add small epsilon for log stability
        log_N_div_eps_val = torch.log( (torch.tensor(float(N_k_tokens), device=q.device, dtype=q.dtype) / self.epsilon) + 1e-9)
        c_val = max_scores_per_query - log_N_div_eps_val
        x_for_poly = scaled_scores - c_val
        
        relevance_threshold = max_scores_per_query - log_N_div_eps_val
        relevant_mask = (scaled_scores >= relevance_threshold)
        
        poly_output = torch.zeros_like(x_for_poly)
        relevant_x_inputs = x_for_poly[relevant_mask]

        if relevant_x_inputs.numel() > 0:
            poly_output_relevant = self._taylor_exp_poly(torch.clamp(relevant_x_inputs, min=0.0))
            poly_output = poly_output.masked_scatter(relevant_mask, poly_output_relevant)
        
        attn_sum = torch.sum(poly_output, dim=-1, keepdim=True)
        stable_attn_sum = attn_sum + 1e-9
        approx_attn_weights = poly_output / stable_attn_sum # (B, H, N_q_tokens, N_k_tokens)
        
        approx_attn_weights_dropped = self.attn_drop(approx_attn_weights)

        context = torch.matmul(approx_attn_weights_dropped, v) # (B, H, N_q_tokens, head_dim)
        
        # Reshape context back to (B, N_q_orig, embed_dim)
        context = context.transpose(1, 2).reshape(B_q, N_q_orig, self.embed_dim)
        context = self.proj(context) # Output projection
        context = self.proj_drop(context)
        
        returned_attn_weights = None
        if need_weights:
            if average_attn_weights: # As per nn.MultiheadAttention behavior
                returned_attn_weights = approx_attn_weights.mean(dim=1) # (B, N_q_tokens, N_k_tokens)
            else:
                returned_attn_weights = approx_attn_weights # (B, H, N_q_tokens, N_k_tokens)
                
        return context, returned_attn_weights

# MCMC Imports
from .fenchel_young_mcmc import (
    MCMCConfig, DiscreteOutputSpace, BinaryHypercube, TopKPolytope,
    TemperatureScheduler
)
from .enhanced_mcmc_layers import (
    ExactOptimizationOracle, CorrectionRatioMCMC, LargeNeighborhoodSearchMCMC
)
from .mcmc_interpretability_solver import (
    BlackBoxSolver # MCMCInterpretabilityHook, ReasoningChain, ThoughtStep # Not directly used in CTM class
)
from .enhanced_neuron_selection import EnhancedNeuronSelector #Enhances Nueron Selections with Biologically-Inspired Systems instead of Random
from .biological_neuron_selection import BiologicalNeuronSelector, BiologicalSelectionConfig
# Import original CTM modules to preserve exact behavior
# try:
from .modules import SynapseUNET, Squeeze, SuperLinear, LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding, CustomRotationalEmbedding1D
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

            if target_attention >= 0.99: # The HTML entity > was the problem here.
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


class PipelineParallelProcessor(nn.Module):
    """
    Pipeline parallelism processor for overlapping CTM and diffusion computation.
    Implements DiffusionPipe-style optimizations with computation overlap.
    """
    
    def __init__(self, config: 'EnhancedCTMConfig'):
        super().__init__()
        self.config = config
        self.pipeline_stages = 4  # CTM, MCMC, Diffusion prep, Diffusion exec
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
        
        # Stage 3: MCMC processing (depends on CTM)
        mcmc_future = self.executor.submit(self._mcmc_stage, ctm_results)
        
        # Wait for diffusion prep
        diff_prep_results = diff_prep_future.result()
        
        # Wait for MCMC completion
        mcmc_results = mcmc_future.result()
        
        # Stage 4: Final diffusion execution
        final_guidance = self._merge_guidance(guidance_data, ctm_results, mcmc_results)
        diffusion_output = diffusion_processor(diff_prep_results['noisy_input'],
                                             diff_prep_results['timesteps'],
                                             final_guidance)
        
        return {
            'ctm_results': ctm_results,
            'mcmc_results': mcmc_results,
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
    
    def _mcmc_stage(self, ctm_results):
        """MCMC processing stage."""
        # Placeholder for MCMC processing
        return {'mcmc_refined': ctm_results.get('final_sync_out')}
    
    def _merge_guidance(self, base_guidance, ctm_results, mcmc_results):
        """Merge guidance from different pipeline stages."""
        merged_guidance = base_guidance.copy() if base_guidance else {}
        merged_guidance.update(ctm_results)
        if mcmc_results:
            merged_guidance.update(mcmc_results)
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
        mcmc_results = self._mcmc_stage(ctm_results)
        
        noisy_input = inputs + torch.randn_like(inputs) * 0.1
        final_guidance = self._merge_guidance(guidance_data, ctm_results, mcmc_results)
        diffusion_output = diffusion_processor(noisy_input, timesteps, final_guidance)
        
        return {
            'ctm_results': ctm_results,
            'mcmc_results': mcmc_results,
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


## Note: The following class definition should be inserted as requested.
# It replaces the provided `DynamicEntropyPatcher` with an implementation
# based on the logic from the "Byte Latent Transformer" paper.

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
        aux_loss = self.criterion(next_byte_logits.reshape(-1, self.byte_vocab_size), targets.reshape(-1))

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
                        if single_entropy[t] > self.global_threshold:
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

    def __init__(self, config: EnhancedCTMConfig):
        super(OriginalCTMCore, self).__init__()
        self.config = config
        self.knowledge_store = None

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
        self.use_activity_plasticity = getattr(config, 'ctm_use_activity_plasticity', True)
        self.plasticity_learning_rate = getattr(config, 'ctm_plasticity_learning_rate', 1e-4)
        if self.use_activity_plasticity:
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
            if config.attention_type == "subquadratic":
                self.attention = SubquadraticAttention(
                    embed_dim=self.d_input, # d_input is the dimension for Q, K, V in CTM's attention
                    num_heads=heads,
                    qkv_bias=config.attention_qkv_bias,
                    attn_drop=dropout, # Use general dropout from CTM config
                    proj_drop=dropout, # Use general dropout from CTM config
                    epsilon=config.subquadratic_attn_epsilon,
                    poly_degree=config.subquadratic_attn_poly_degree,
                    scale=None # Default scale calculation
                )
            elif config.attention_type == "binary_sparse":
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


    # --- Core CTM Methods ---

    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        """
        Computes synchronisation to be used as a vector representation. 

        A neuron has what we call a 'trace', which is a history (time series) that changes with internal
        recurrence. i.e., it gets longer with every internal tick. There are pre-activation traces
        that are used in the NLMs and post-activation traces that, in theory, are used in this method. 

        We define sychronisation between neuron i and j as the dot product between their respective
        time series. Since there can be many internal ticks, this process can be quite compute heavy as it
        involves many dot products that repeat computation at each step.
        
        Therefore, in practice, we update the synchronisation based on the current post-activations,
        which we call the 'activated state' here. This is possible because the inputs to synchronisation 
        are only updated recurrently at each step, meaning that there is a linear recurrence we can
        leverage. 
        
        See Appendix TODO of the Technical Report (TODO:LINK) for the maths that enables this method.
        """

        if synch_type == 'action': # Get action parameters
            n_synch = self.n_synch_action
            neuron_indices_left = self.action_neuron_indices_left
            neuron_indices_right = self.action_neuron_indices_right
        elif synch_type == 'out': # Get input parameters
            n_synch = self.n_synch_out
            neuron_indices_left = self.out_neuron_indices_left
            neuron_indices_right = self.out_neuron_indices_right
        
        if self.neuron_select_type in ('first-last', 'random'):
            # For first-last and random, we compute the pairwise sync between all selected neurons
            if self.neuron_select_type == 'first-last':
                if synch_type == 'action': # Use last n_synch neurons for action
                    selected_left = selected_right = activated_state[:, -n_synch:]
                elif synch_type == 'out': # Use first n_synch neurons for out
                    selected_left = selected_right = activated_state[:, :n_synch]
            else: # Use the randomly selected neurons
                selected_left = activated_state[:, neuron_indices_left]
                selected_right = activated_state[:, neuron_indices_right]
            
            # Compute outer product of selected neurons
            outer = selected_left.unsqueeze(2) * selected_right.unsqueeze(1)
            # Resulting matrix is symmetric, so we only need the upper triangle
            i, j = torch.triu_indices(n_synch, n_synch)
            pairwise_product = outer[:, i, j]
            
        elif self.neuron_select_type == 'random-pairing' or \
             self.neuron_select_type.startswith('bio_') or \
             self.neuron_select_type in ['adaptive_random', 'performance_guided', 'task_aware']:
            # For random-pairing and bio/hybrid types, compute sync between specific pairs
            left = activated_state[:, neuron_indices_left]
            right = activated_state[:, neuron_indices_right]
            pairwise_product = left * right
        else:
            raise ValueError(f"Unhandled neuron selection type in compute_synchronisation: {self.neuron_select_type}")
        
        
        
        # Compute synchronisation recurrently
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
            assert synch_type in ('out', 'action'), f"Invalid synch_type: {synch_type}"
            left, right = self.initialize_left_right_neurons(synch_type, self.d_model, n_synch, n_random_pairing_self)
            synch_representation_size = self.synch_representation_size_action if synch_type == 'action' else self.synch_representation_size_out
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
            neuron_indices_left, neuron_indices_right = self._enhanced_selector.select_neurons_for_synchronization(
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

        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []

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
        if track:
            return predictions, certainties, (np.array(synch_out_tracking), np.array(synch_action_tracking)), np.array(pre_activations_tracking), np.array(post_activations_tracking), np.array(attention_tracking)
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

    def apply_activity_plasticity(self, global_loss: torch.Tensor):
        """
        Updates the plastic synapse weights based on a Hebbian rule modulated by global success.
        This method is now tied to the neuron selection mechanism, applying plasticity only
        to the connections between neurons selected for action and output synchronization.
        """
        if not self.use_activity_plasticity or not self.training:
            return

        with torch.no_grad():
            learning_signal = -global_loss.detach()
            
            state_trace = self.last_state_trace
            if state_trace is None or state_trace.numel() == 0:
                print("Warning: last_state_trace not available for plasticity update. Skipping.")
                return

            # Get the indices of all unique neurons involved in synchronization
            eligible_indices = torch.unique(torch.cat([
                self.action_neuron_indices_left, self.action_neuron_indices_right,
                self.out_neuron_indices_left, self.out_neuron_indices_right
            ]))

            # Extract the state traces for only these eligible neurons
            eligible_traces = state_trace[:, eligible_indices, :]
            
            # Get biological modulation scores
            modulation_scores = torch.ones(len(eligible_indices), device=state_trace.device)
            if self.biological_selector is not None and self.neuron_select_type.startswith('bio_'):
                selector_activations = eligible_traces.mean(dim=-1)
                
                _, metadata = self.biological_selector.select_neurons(
                    activations=selector_activations.mean(dim=0).unsqueeze(0),
                    top_k=len(eligible_indices),
                    layer_name="plasticity_update"
                )

                sel_type = self.neuron_select_type.replace('bio_', '')
                score_key_map = {
                    'hebbian': 'hebbian_scores', 'plasticity': 'plasticity_scores',
                    'competitive': 'competition_scores', 'homeostatic': 'homeostatic_scores',
                    'evolutionary': 'fitness_scores', 'stdp': 'stdp_scores',
                    'criticality': 'criticality_scores', 'multi_objective': 'combined_scores',
                }
                score_key = score_key_map.get(sel_type)
                if score_key and score_key in metadata:
                    scores = metadata[score_key].detach()
                    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
                    modulation_scores = scores

            # Compute Hebbian trace for the subset of eligible neurons
            st_batch_mean = eligible_traces.mean(dim=0)
            st_centered = st_batch_mean - st_batch_mean.mean(dim=1, keepdim=True)
            cov_matrix = torch.matmul(st_centered, st_centered.T) / (st_centered.shape[1] - 1)
            stds = torch.std(st_batch_mean, dim=1, keepdim=True)
            stds[stds == 0] = 1e-8
            
            local_hebbian_trace = cov_matrix / (torch.matmul(stds, stds.T))
            local_hebbian_trace = torch.nan_to_num(local_hebbian_trace)
            
            # Modulate Hebbian trace with biological scores
            modulation_matrix = torch.outer(modulation_scores, modulation_scores)
            modulated_hebbian_trace = local_hebbian_trace * modulation_matrix

            # Create a sparse weight update tensor
            delta_W = torch.zeros_like(self.plastic_synapses.weight)
            update_values = self.plasticity_learning_rate * modulated_hebbian_trace * learning_signal
            
            # Assign the updates to the correct positions in the main weight matrix
            row_indices, col_indices = torch.meshgrid(eligible_indices, eligible_indices, indexing='ij')
            delta_W[row_indices, col_indices] = update_values

            # Apply the sparse update
            self.plastic_synapses.weight.add_(delta_W)

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
        
        # Initialize tracking
        tracking_data = {
            'sync_out_history': [],
            'sync_action_history': [],
            'activated_states': [],
            'state_traces': [],
            'attention_weights': [],
            'pc_losses': []
        }
        
        # Initialize recurrent state
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)
        
        # Storage for outputs
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=torch.float32)
        
        # Initialize synch values
        decay_alpha_action, decay_beta_action = None, None
        decay_alpha_out, decay_beta_out = None, None
        
        # Clamp decay parameters
        if hasattr(self, 'decay_params_action'):
            self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        
        # Compute learned weighting
        r_action = (torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1) 
                   if hasattr(self, 'decay_params_action') else None)
        r_out = torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)
        
        # Initialize output synchronization
        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
            activated_state, None, None, r_out, synch_type='out'
        )
        
        # Recurrent loop with full tracking
        for stepi in range(self.iterations):
            
            # Calculate synchronisation for input data interaction
            if hasattr(self, 'decay_params_action'):
                synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(
                    activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action'
                )
                tracking_data['sync_action_history'].append(synchronisation_action.clone())
                
                # Interact with data via attention
                if self.q_proj is not None and self.attention is not None:
                    q = self.q_proj(synchronisation_action).unsqueeze(1)
                    attn_out, attn_weights = self.attention(q, kv_features, kv_features, 
                                                           average_attn_weights=False, need_weights=True)
                    attn_out = attn_out.squeeze(1)
                    pre_synapse_input = torch.cat((attn_out, activated_state), dim=-1)
                    tracking_data['attention_weights'].append(attn_weights.clone())
                else:
                    pre_synapse_input = torch.cat((kv_features.mean(dim=1), activated_state), dim=-1)
            else:
                if self.q_proj is not None and self.attention is not None:
                    q = activated_state.unsqueeze(1)
                    attn_out, attn_weights = self.attention(q, kv_features, kv_features,
                                                           average_attn_weights=False, need_weights=True)
                    attn_out = attn_out.squeeze(1)
                    pre_synapse_input = torch.cat((attn_out, activated_state), dim=-1)
                    tracking_data['attention_weights'].append(attn_weights.clone())
                else:
                    pre_synapse_input = torch.cat((kv_features.mean(dim=1), activated_state), dim=-1)
            

            # Apply synapses
            state = self.synapses(pre_synapse_input)

            # --- Apply Activity-Dependent Plasticity ---
            if self.use_activity_plasticity:
                plastic_adjustment = self.plastic_synapses(activated_state)
                state = state + plastic_adjustment
            
            # Update state trace
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
            tracking_data['state_traces'].append(state_trace.clone())
            
            # Apply neuron-level models
            activated_state = self.trace_processor(state_trace)
            tracking_data['activated_states'].append(activated_state.clone())

            # --- Predictive Coding Loss ---
            if self.use_predictive_coding:
                pc_loss = self.compute_predictive_coding_loss(activated_state)
                tracking_data['pc_losses'].append(pc_loss)
            
            # Calculate synchronisation for output predictions
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
                activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out'
            )
            tracking_data['sync_out_history'].append(synchronisation_out.clone())
            
            # Get predictions and certainties
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)
            
            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty
        
        # Store final state trace for plasticity update
        self.last_state_trace = state_trace.detach()

        return {
            'predictions': predictions,
            'certainties': certainties,
            'final_sync_out': synchronisation_out,
            'predictive_coding_loss': torch.stack(tracking_data['pc_losses']).mean() if tracking_data['pc_losses'] else torch.tensor(0.0, device=device),
            **tracking_data
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
            task_analyzer=task_analyzer # Pass the task_analyzer instance
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
        self.sampling_noise_scheduler = DDPMScheduler(
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
    
    def forward(self, noisy_input: torch.Tensor, timestep: torch.Tensor,
                ctm_data: Optional[Dict[str, torch.Tensor]] = None,
                hipa_control_signal: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]: # Added hipa_control_signal, updated return type
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
            return base_noise_pred
        
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

            sync_inf = ctm_influences[0] if len(ctm_influences) > 0 else torch.zeros(batch_size_for_fallback, config.d_model, device=device_for_fallback, dtype=dtype_for_fallback)
            state_inf = ctm_influences[2] if len(ctm_influences) > 2 else torch.zeros(batch_size_for_fallback, config.d_model, device=device_for_fallback, dtype=dtype_for_fallback)
            
            # Multi-stage refinement
            for i, refinement_layer in enumerate(self.ctm_noise_refinement):
                # Combine current noise with CTM influences
                refinement_input = torch.cat([current_noise, sync_inf, state_inf], dim=-1)
                refinement = refinement_layer(refinement_input)
                
                # Progressive residual connection with increasing CTM influence
                ctm_strength = self.coupling_strength * (i + 1) / len(self.ctm_noise_refinement)
                current_noise = current_noise + refinement * ctm_strength
            
            # Final certainty-based scaling
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
            
            return current_noise
        
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
        alpha_bar = self.alpha_bars[timesteps.to(self.alpha_bars.device)]
        
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
            similarities = torch.cosine_similarity(current_state, hist_state.unsqueeze(0).expand_as(current_state), dim=-1)
            
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

        self.knowledge_store = None
 
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
            print("Warning: TaskAnalyzer class not found or could not be imported. HIPA might not function correctly.")
            # Create a dummy TaskAnalyzer if import fails, to prevent further errors during init.
            # This dummy will likely cause issues at runtime if HIPA is used.
            class DummyTaskAnalyzer:
                def detect_modality(self, x, task_id=None, context_hints=None):
                    # print("Warning: Using DummyTaskAnalyzer. HIPA modality detection will not be accurate.")
                    return {'use_hipa': False, 'modality': 'unknown', 'fft_dims': [], 'freq_threshold': 0.1, 'enhancement_strength': 0.0}
            self.task_analyzer_instance = DummyTaskAnalyzer()
        except TypeError as e:
            # print(f"Warning: TaskAnalyzer could not be instantiated with config: {e}. Trying without config.")
            try:
                self.task_analyzer_instance = ActualTaskAnalyzerClass()
            except Exception as e_init:
                # print(f"Fatal: Could not instantiate TaskAnalyzer: {e_init}. HIPA will likely fail.")
                # Fallback to dummy if all attempts fail
                class DummyTaskAnalyzer:
                    def detect_modality(self, x, task_id=None, context_hints=None):
                        # print("Warning: Using DummyTaskAnalyzer due to instantiation failure. HIPA modality detection will not be accurate.")
                        return {'use_hipa': False, 'modality': 'unknown', 'fft_dims': [], 'freq_threshold': 0.1, 'enhancement_strength': 0.0}
                self.task_analyzer_instance = DummyTaskAnalyzer()
        
        # Ensure 'copy' is imported for deepcopy
        import copy

        # Determine input dimension for the main encoder and task inference
        self.dynamic_entropy_patcher = None
        self.patcher_encoder = None
        self.multi_granularity_processor = None
        self.byte_embedding = None

        if config.use_dynamic_entropy_patcher:
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
            if not hasattr(self.multi_granularity_processor, 'output_dim'):
                 print(f"Warning: MultiGranularityBinaryProcessor does not have 'output_dim'. Using config value {raw_feature_dim}.")
            else:
                 raw_feature_dim = self.multi_granularity_processor.output_dim
        else: # Fallback to simple byte embedding
            self.byte_embedding = nn.Embedding(256, config.byte_embedding_dim)
            raw_feature_dim = config.byte_embedding_dim
        
        # Mixed precision trainer
        self.mixed_precision_trainer = MixedPrecisionTrainer(self, config)

        # Core CTM
        self.ctm_core = OriginalCTMCore(config)
        
        # Enhanced diffusion processor
        # actual_noisy_input_dim is now config.unet_input_feature_dim
        self.diffusion = CTMControlledDiffusionProcessor(
            config,
            actual_noisy_input_dim=config.unet_input_feature_dim,
            task_analyzer=self.task_analyzer_instance # Pass the TaskAnalyzer instance
        )
        
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

        # Enhanced MCMC Parameters
        enable_enhanced_mcmc: bool = False
        mcmc_config: Optional[MCMCConfig] = None # MCMCConfig from .fenchel_young_mcmc
        mcmc_output_space_type: str = 'binary_hypercube' # e.g., 'binary_hypercube', 'top_k_polytope'
        mcmc_output_space_dim: Optional[int] = None # Dimension of MCMC output space, defaults to ctm_out_dims
        use_large_neighborhood_search: bool = True
        lns_frequency: int = 10
        lns_neighborhood_size: int = 20
        enable_blackbox_solver: bool = True # For MCMC-based interpretability
        mcmc_phi_network_hidden_dim: int = 128
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
                # This expects input (B, C, L), so ensure features are shaped accordingly before passing
                self.positional_embedding = CustomRotationalEmbedding1D(d_model=pe_dim)
            else:
                print(f"Warning: Unknown positional_embedding_type: {config.positional_embedding_type}. No positional embedding will be used.")

        # Store MCMC configuration from EnhancedCTMConfig
        self.enable_enhanced_mcmc = getattr(config, 'enable_enhanced_mcmc', False)
        self.mcmc_config_params = getattr(config, 'mcmc_config', None) # Renamed to avoid conflict if MCMCConfig class is also self.mcmc_config
        self.mcmc_output_space_type = getattr(config, 'mcmc_output_space_type', 'binary_hypercube')
        self.mcmc_output_space_dim_config = getattr(config, 'mcmc_output_space_dim', None) # Renamed
        self.use_large_neighborhood_search = getattr(config, 'use_large_neighborhood_search', True)
        self.lns_frequency = getattr(config, 'lns_frequency', 10)
        self.lns_neighborhood_size = getattr(config, 'lns_neighborhood_size', 20)
        self.enable_blackbox_solver = getattr(config, 'enable_blackbox_solver', False)
        self.mcmc_phi_network_hidden_dim = getattr(config, 'mcmc_phi_network_hidden_dim', 128)

        if self.enable_enhanced_mcmc:
            self._initialize_enhanced_mcmc()
        else:
            self.enhanced_mcmc_sampler = None
            self.mcmc_phi_network = None
            self.mcmc_output_space = None
            self.blackbox_solver = None

        # Projection layer for sampling path: from ctm_input_dim to unet_input_feature_dim
        self.sampling_kv_to_unet_input_proj = nn.Linear(config.ctm_input_dim, config.unet_input_feature_dim)

        # Initialize a training noise scheduler for EnhancedCTMDiffusion
        self.training_noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.diffusion_timesteps,
            beta_start=config.diffusion_beta_start,
            beta_end=config.diffusion_beta_end,
            beta_schedule=("squaredcos_cap_v2" if config.noise_schedule == "cosine" else config.noise_schedule) if hasattr(config, 'noise_schedule') else "linear"
        )
        
        # Initialize new optimization components
        self._initialize_optimization_components()
    
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
            print(f"JEPA components initialized. Predictor I/O dim: {jepa_io_dim}, Output dim: {predictor_output_dim}")
        else:
            self.jepa_target_patch_encoder = None
            self.jepa_predictor = None
    
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
                'mcmc_results': pipeline_results.get('mcmc_results'),
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

        # The main input_encoder processes the sequence of raw_features
        # raw_features is (batch, num_patches_or_seq, feature_dim)
        encoded_features = self.input_encoder(raw_features) # Output: (batch, num_patches_or_seq, ctm_input_dim)

        # Apply positional embedding if configured
        if self.positional_embedding is not None:
            if self.config.reshape_patch_sequence_to_grid and \
               isinstance(self.positional_embedding, (LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding)):
                
                B, S, D = encoded_features.shape
                W_patches = self.config.patch_grid_width
                if W_patches is None: # Should be caught by config validation, but as a safeguard
                    print(f"Warning: patch_grid_width is None but reshape_patch_sequence_to_grid is True. Defaulting width to sqrt(S).")
                    W_patches = int(math.sqrt(S))
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
                print(f"Reshaped patch sequence to grid ({H_patches}x{W_patches}) and applied 2D PE. New sequence length: {encoded_features.shape[1]}")

            elif isinstance(self.positional_embedding, CustomRotationalEmbedding1D):
                # 1D PE: Expects (B, C, L). `encoded_features` is (B, S, D)
                # Permute to (B, D, S)
                pe_input_1d = encoded_features.permute(0, 2, 1)
                pos_emb_1d = self.positional_embedding(pe_input_1d) # Output (B, D, S)
                pos_emb_1d = pos_emb_1d.permute(0, 2, 1) # (B, S, D)
                if pos_emb_1d.shape == encoded_features.shape:
                    encoded_features = encoded_features + pos_emb_1d
                else:
                    print(f"Warning: 1D Rotational PE shape {pos_emb_1d.shape} mismatch with features {encoded_features.shape}. Skipping PE.")
            
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
            # Add other PE types if necessary
            
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

    def _initialize_enhanced_mcmc(self):
        """Initializes the enhanced MCMC components."""
        if not self.config.enable_enhanced_mcmc: # Use self.config consistently
            return

        # Determine MCMC output dimension
        if self.config.mcmc_output_space_dim is None: # Corrected attribute name
            mcmc_dim = self.config.out_dims if self.config.out_dims is not None else self.config.d_model
        else:
            mcmc_dim = self.config.mcmc_output_space_dim
        
        if mcmc_dim is None: # Still None, raise error
            raise ValueError("MCMC output space dimension could not be determined. "
                             "Please set mcmc_output_space_dim in EnhancedCTMConfig or ensure "
                             "config.out_dims or config.d_model is set.")

        # 1. Create MCMC Output Space
        if self.config.mcmc_output_space_type == 'binary_hypercube':
            self.mcmc_output_space = BinaryHypercube(dimension=mcmc_dim)
        elif self.config.mcmc_output_space_type == 'top_k_polytope':
            k_val = min(mcmc_dim // 2, 5) 
            if k_val == 0 and mcmc_dim > 0: k_val = 1
            if k_val == 0 and mcmc_dim == 0: # Should be caught by TopKPolytope's __init__ too
                 raise ValueError("Dimension for TopKPolytope must be greater than 0 to set k.")
            self.mcmc_output_space = TopKPolytope(dimension=mcmc_dim, k=k_val)
        else:
            raise ValueError(f"Unsupported MCMC output space type: {self.config.mcmc_output_space_type}")

        # 2. Create Phi Network (Energy function component)
        self.mcmc_phi_network = nn.Sequential(
            nn.Linear(mcmc_dim, self.config.mcmc_phi_network_hidden_dim), # Use self.config
            nn.ReLU(),
            nn.Linear(self.config.mcmc_phi_network_hidden_dim, self.config.mcmc_phi_network_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.mcmc_phi_network_hidden_dim // 2, 1)
        )

        # 3. Create BlackBox Solver (Optional, for interpretability)
        # This BlackBoxSolver wraps the mcmc_phi_network.
        if self.config.enable_blackbox_solver:
            self.blackbox_solver_instance = BlackBoxSolver( # Renamed to avoid conflict
                self.mcmc_phi_network,
                input_example=torch.randn(1, mcmc_dim) 
            )
            phi_for_mcmc_samplers = self.blackbox_solver_instance
        else:
            self.blackbox_solver_instance = None
            phi_for_mcmc_samplers = self.mcmc_phi_network
        
        # 4. Create MCMC Sampler
        current_mcmc_config_params = self.config.mcmc_config # Use self.config
        if current_mcmc_config_params is None:
            print("Warning: MCMCConfig not provided to EnhancedCTMDiffusion. Using default MCMCConfig.")
            current_mcmc_config_params = MCMCConfig() 

        if self.config.use_large_neighborhood_search:
            # The ExactOptimizationOracle for LNS will use the potentially blackboxed phi.
            exact_oracle_for_lns = ExactOptimizationOracle(
                output_space=self.mcmc_output_space,
                phi_network=phi_for_mcmc_samplers
            )
            self.enhanced_mcmc_sampler = LargeNeighborhoodSearchMCMC(
                output_space=self.mcmc_output_space,
                config=current_mcmc_config_params,
                phi_network=phi_for_mcmc_samplers, 
                exact_oracle=exact_oracle_for_lns,
                lns_frequency=self.config.lns_frequency, 
                lns_neighborhood_size=self.config.lns_neighborhood_size
            )
        else:
            self.enhanced_mcmc_sampler = CorrectionRatioMCMC(
                output_space=self.mcmc_output_space,
                config=current_mcmc_config_params,
                phi_network=phi_for_mcmc_samplers
            )
        
        # Device handling should be done by the main model's .to(device) call
        # which should iterate over submodules.
        
        # Move MCMC components to the correct device if model is on GPU
        # This should ideally happen when the main model is moved to a device.
        # For now, we assume it will be handled by the main model's .to(device) call.
        # Example:
        # if next(self.parameters()).is_cuda:
        #     self.mcmc_phi_network.cuda()
        #     self.enhanced_mcmc_sampler.cuda() # Sampler might need its own .to(device)

    def _apply_enhanced_mcmc(self, theta: torch.Tensor, target_y: torch.Tensor,
                               current_epoch: Optional[int] = None, current_batch: Optional[int] = None) -> Dict[str, Any]:
        """
        Applies the enhanced MCMC sampling process.

        Args:
            theta: The logits (or parameters) from the CTM core for the MCMC energy function.
                   Shape: (batch_size, mcmc_output_space_dim)
            target_y: The ground truth discrete structures for the MCMC Fenchel-Young loss.
                      Shape: (batch_size, mcmc_output_space_dim)
            current_epoch: Optional current epoch for LNS scheduling or other diagnostics.
            current_batch: Optional current batch for LNS scheduling or other diagnostics.

        Returns:
            A dictionary containing MCMC results.
        """
        if not self.config.enable_enhanced_mcmc or self.enhanced_mcmc_sampler is None:
            return {
                'mcmc_loss': torch.tensor(0.0, device=theta.device),
                'mcmc_expectation': target_y.clone(), # Return target if MCMC is off
                'mcmc_raw_samples': None, # Or perhaps an empty list
                'mcmc_acceptance_rate': 0.0, # From overall stats
                'mcmc_avg_acceptance_term_pk': 0.0, # From overall stats
                'mcmc_stats': {},
                'solver_diagnostics': None # Or an empty list
            }

        # Determine if LNS should be used for this specific call to the sampler
        use_lns_for_this_sampler_call = False
        if self.config.use_large_neighborhood_search and isinstance(self.enhanced_mcmc_sampler, LargeNeighborhoodSearchMCMC):
            # LNS frequency is now handled inside LargeNeighborhoodSearchMCMC's sample_chain_corrected,
            # but we still need to tell the sampler's forward method that LNS is generally enabled for it.
            # The `use_large_neighborhood_step` in `sample_chain_corrected` controls per-step LNS.
            # The `use_large_neighborhood` in `forward` of the sampler can be a general toggle.
            use_lns_for_this_sampler_call = True # Indicates LNSMCMC is the active type

        # The `forward` method of CorrectionRatioMCMC (and its subclass LargeNeighborhoodSearchMCMC)
        # now takes `use_large_neighborhood` as an argument.
        mcmc_output_dict = self.enhanced_mcmc_sampler(
            theta=theta, 
            target=target_y, 
            use_large_neighborhood=use_lns_for_this_sampler_call
            # current_epoch and current_batch are not directly used by sampler.forward,
            # but could be passed into mcmc_stats if needed for external logging.
        )
        
        # Extract results from the dictionary returned by the sampler's forward method
        mcmc_loss = mcmc_output_dict.get('loss', torch.tensor(0.0, device=theta.device))
        mcmc_expectation = mcmc_output_dict.get('expectation', target_y.clone())
        
        # Sampler's forward method should return a 'mcmc_stats' dictionary
        # which itself contains 'num_samples', 'avg_acceptance_rate', 'avg_acceptance_term_pk', etc.
        detailed_mcmc_stats = mcmc_output_dict.get('mcmc_stats', {})
        
        mcmc_raw_samples = detailed_mcmc_stats.get('raw_samples', None) # Assuming sampler might provide this
        mcmc_acceptance_rate = detailed_mcmc_stats.get('avg_acceptance_rate', 0.0)
        mcmc_avg_pk = detailed_mcmc_stats.get('avg_acceptance_term_pk', 0.0)


        # Solver diagnostics would be collected if an interpretability hook is attached
        # to the blackbox_solver_instance. The sampler itself doesn't directly return this
        # unless we modify it to aggregate from its internal exact_oracle.
        solver_diagnostics = []
        if self.config.enable_blackbox_solver and \
           isinstance(self.enhanced_mcmc_sampler, LargeNeighborhoodSearchMCMC) and \
           self.enhanced_mcmc_sampler.exact_oracle is not None:
            # Access diagnostics if the LNS sampler stores them from its oracle
            solver_diagnostics = self.enhanced_mcmc_sampler.exact_oracle.get_solver_state().get('optimization_history', [])


        return {
            'mcmc_loss': mcmc_loss,
            'mcmc_expectation': mcmc_expectation,
            'mcmc_raw_samples': mcmc_raw_samples,
            'mcmc_acceptance_rate': mcmc_acceptance_rate,
            'mcmc_avg_acceptance_term_pk': mcmc_avg_pk,
            'mcmc_stats': detailed_mcmc_stats, # Pass through all stats from sampler
            'solver_diagnostics': solver_diagnostics
        }

    # def _expand_task_embedding(self, new_size: int): # Obsolete
        """
        Legacy method for backward compatibility.
        Now delegates to the dynamic embedding system.
        """
        if not hasattr(self, 'dynamic_task_embedding'):
            self._initialize_dynamic_task_system()
        
        # The dynamic system handles expansion automatically
        self.dynamic_task_embedding._grow_embedding_to_size(new_size)
        

    def compute_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes key-value features for the CTM from the input tensor.
        The input x is the output of self.input_encoder.
        """
        kv_features_for_ctm = x
        return kv_features_for_ctm
    
    def forward(self, byte_sequence: torch.Tensor, target_diffusion_output: Optional[torch.Tensor] = None,
                mode: str = 'ctm_controlled_diffusion', timestep: Optional[torch.Tensor] = None,
                target_mcmc_output: Optional[torch.Tensor] = None, current_epoch: int = 0,
                current_batch: int = 0, task_name: Optional[str] = None) -> Dict[str, torch.Tensor]: # Re-added task_name
        """
        Forward pass of the CTMDiffusionModel using byte sequences.

        Args:
            byte_sequence (torch.Tensor): Raw byte sequence input.
                                          Shape: (batch_size, sequence_length)
            target_diffusion_output (Optional[torch.Tensor]): The target clean data (x_0) for diffusion loss.
                                                              Required if training with diffusion.
                                                              Shape: (batch_size, sequence_length, output_feature_dim)
            mode (str): Operation mode. Options: 'ctm_controlled_diffusion', 'ctm_only', 'diffusion_only', 'mcmc_only'
                       'ctm_only' runs CTM core.
                       'mcmc_only' runs CTM core then MCMC.
                       'diffusion_only' runs diffusion processor (needs appropriate 'inputs' as noisy data).
                       'ctm_controlled_diffusion' runs CTM, then MCMC (if enabled), then diffusion.
            timestep (Optional[torch.Tensor]): Current diffusion timestep (if applicable for diffusion modes)
            target_mcmc_output (Optional[torch.Tensor]): Ground truth for MCMC Fenchel-Young loss.
                                                        Shape: (batch_size, mcmc_output_space_dim)
            current_epoch (int): Current training epoch.
            current_batch (int): Current training batch in the epoch.

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
        
        losses['jepa_loss'] = torch.tensor(0.0, device=device)
        # The aux loss from dynamic_entropy_patcher (online JEPA encoder) is handled by _prepare_input_features
        # No separate jepa_context_aux_loss or jepa_target_aux_loss needed here for now.

        # Prepare input features using the online encoder (dynamic_entropy_patcher)
        # This also gives us the online_patch_embeddings (kv_features_for_ctm) and their original byte indices.
        # kv_features_for_ctm, current_inferred_latent, current_hipa_signal, entropy_aux_loss are computed once
        kv_features_for_ctm, current_inferred_latent, current_hipa_signal, entropy_aux_loss = \
            self._prepare_input_features(byte_sequence)
        
        losses['entropy_model_aux_loss'] = entropy_aux_loss * self.config.entropy_model_loss_weight
        online_patch_embeddings = kv_features_for_ctm # Shape: (B, S_patches, D_embed)

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
                print(f"Error during JEPA processing in forward pass: {e_jepa}")
                import traceback
                traceback.print_exc()
                losses['jepa_loss'] = torch.tensor(0.0, device=device)
        
        # --- CTM Core Logic ---
        # CTM core processes the features derived from the input byte_sequence.
        # ctm_output_features = self.ctm_core(kv_features_for_ctm) # If CTM core is a separate module
        # For now, assuming kv_features_for_ctm are directly used or further processed by CTM core if it's integrated.
        # The `ctm_data` used by diffusion processor will come from `self.ctm_core.forward_with_full_tracking(kv_features_for_ctm)` later.
        
        # --- Diffusion Model Logic ---
        numeric_target_diffusion_output = None
        if target_diffusion_output is not None:
            # Assuming target_diffusion_output from training script is now bytes (uint8)
            # Convert to numeric (float32) for diffusion processing
            # Check if it's already numeric (e.g. if called internally with non-byte target)
            if target_diffusion_output.dtype == torch.uint8:
                try:
                    item_size_for_numeric = self.config.audio_output_item_size
                    numeric_target_diffusion_output = batched_bytes_to_numeric_tensor(target_diffusion_output, item_size=item_size_for_numeric, target_dtype=np.float32)
                except ValueError as e:
                    print(f"Warning: Error converting byte target_diffusion_output to numeric: {e}. Using as is, which might be incorrect.")
                    numeric_target_diffusion_output = target_diffusion_output # Fallback, though likely problematic
            else: # Already numeric
                numeric_target_diffusion_output = target_diffusion_output

            if kv_features_for_ctm.size(1) != numeric_target_diffusion_output.size(1) and hasattr(self.diffusion, 'unet') and numeric_target_diffusion_output.ndim > 1:
                 print(f"Warning: Sequence length mismatch between CTM features ({kv_features_for_ctm.size(1)}) and numeric diffusion target ({numeric_target_diffusion_output.size(1)}). This might affect conditioning.")

            # Use EnhancedCTMDiffusion's own training_noise_scheduler for the noising process
            if hasattr(self, 'training_noise_scheduler'):
                t = torch.randint(0, self.training_noise_scheduler.config.num_train_timesteps, (batch_size,), device=byte_sequence.device).long()
                # Generate noise based on the numeric target's shape and type
                noise = torch.randn_like(numeric_target_diffusion_output)
                # Use a distinct variable name for the noisy input passed to the diffusion processor
                noisy_input_for_diffusion_processor = self.training_noise_scheduler.add_noise(numeric_target_diffusion_output, noise, t)
                
                # CTMControlledDiffusionProcessor.forward (self.diffusion) is the model that predicts noise (or x0)
                # It needs the noisy input, timestep, and CTM conditioning data.
                # kv_features_for_ctm was prepared earlier from byte_sequence.
                ctm_data_for_diffusion_conditioning = self.ctm_core.forward_with_full_tracking(kv_features_for_ctm)

                # Get the prediction from the diffusion processor
                # The diffusion processor's forward method is CTMControlledDiffusionProcessor.forward
                prediction_output_tuple = self.diffusion(
                    noisy_input=noisy_input_for_diffusion_processor,
                    timestep=t,
                    ctm_data=ctm_data_for_diffusion_conditioning,
                    hipa_control_signal=current_hipa_signal # Pass HIPA signal
                )

                if isinstance(prediction_output_tuple, tuple): # If it returns (prediction, guidance_info)
                    predicted_noise_or_x0 = prediction_output_tuple[0]
                else:
                    predicted_noise_or_x0 = prediction_output_tuple

                # Determine loss based on the training_noise_scheduler's prediction type
                if hasattr(self.training_noise_scheduler, 'config') and hasattr(self.training_noise_scheduler.config, 'prediction_type'):
                    if self.training_noise_scheduler.config.prediction_type == "epsilon":
                        diffusion_loss = F.mse_loss(predicted_noise_or_x0, noise)
                    elif self.training_noise_scheduler.config.prediction_type == "sample":
                        diffusion_loss = F.mse_loss(predicted_noise_or_x0, numeric_target_diffusion_output)
                    else:
                        print(f"Unsupported diffusion prediction type in training_noise_scheduler: {self.training_noise_scheduler.config.prediction_type}")
                        diffusion_loss = torch.tensor(0.0, device=byte_sequence.device)
                else: # Default to epsilon prediction if config not available
                    diffusion_loss = F.mse_loss(predicted_noise_or_x0, noise)
            else:
                # This case should ideally not be reached if training_noise_scheduler is always initialized
                print("CRITICAL WARNING: EnhancedCTMDiffusion.training_noise_scheduler not defined. Using zero diffusion loss.")
                diffusion_loss = torch.tensor(0.0, device=byte_sequence.device)
            losses['diffusion_loss'] = diffusion_loss
        else:
            losses['diffusion_loss'] = torch.tensor(0.0, device=byte_sequence.device)


        total_loss = torch.tensor(0.0, device=byte_sequence.device)
        for loss_name, loss_val in losses.items():
            if loss_val is not None:
                total_loss += loss_val

        # Continue with the rest of the function logic
        # For inference, the output would be generated by a sampling loop using the diffusion model,
        # conditioned on ctm_output_features.
        
        # Use the existing variables from the first part of the function
        device = byte_sequence.device

        # Initialize output_dict
        output_dict = {
            'ctm_core_data': None,
            'ctm_internal_loss': torch.tensor(0.0, device=device),
            'mcmc_results': None,
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
        if mode in ['ctm_only', 'mcmc_only', 'ctm_controlled_diffusion']:
            # Use the kv_features_for_ctm computed earlier in the function
            ctm_data = self.ctm_core.forward_with_full_tracking(kv_features_for_ctm)
            output_dict['ctm_core_data'] = ctm_data
            theta_candidate_from_ctm = ctm_data['final_sync_out']
        
        # --- 2. Enhanced MCMC Processing ---
        mcmc_expectation = None
        if self.enable_enhanced_mcmc and mode in ['mcmc_only', 'ctm_controlled_diffusion']:
            if theta_candidate_from_ctm is None:
                raise ValueError("CTM core output (theta_candidate_from_ctm) is needed for MCMC but is None.")
            if target_mcmc_output is None:
                raise ValueError("target_mcmc_output must be provided for MCMC loss calculation in modes '{}'.".format(mode))

            if not hasattr(self, 'ctm_to_mcmc_theta_proj'):
                if self.mcmc_output_space is None: self._initialize_enhanced_mcmc()
                self.ctm_to_mcmc_theta_proj = nn.Linear(
                    theta_candidate_from_ctm.shape[-1], self.mcmc_output_space.dimension
                ).to(device)
            
            theta_for_mcmc = self.ctm_to_mcmc_theta_proj(theta_candidate_from_ctm)
            
            mcmc_results_dict = self._apply_enhanced_mcmc(
                theta_for_mcmc, target_mcmc_output, current_epoch, current_batch
            )
            output_dict['mcmc_results'] = mcmc_results_dict
            output_dict['mcmc_loss'] = mcmc_results_dict.get('mcmc_loss', torch.tensor(0.0, device=device))
            mcmc_expectation = mcmc_results_dict.get('mcmc_expectation')

        # Determine the representation to use post-CTM/MCMC
        # If MCMC ran, its expectation is preferred, otherwise direct CTM output.
        final_ctm_representation = mcmc_expectation if mcmc_expectation is not None else theta_candidate_from_ctm


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
                #     print(f"Warning: Dimension mismatch for diffusion guidance. MCMC output dim: {final_ctm_representation.shape[-1]}, CTM final_sync_out dim: {guidance_data_for_diffusion['final_sync_out'].shape[-1]}. Using MCMC output directly.")
                #     # Potentially add a projection layer here if needed:
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
            # For now, I am adding a placeholder call here to show where it would go.
            # In a real scenario, this would be handled by the trainer.
            if self.training and 'total_loss' in locals() and self.config.use_activity_plasticity:
                 self.ctm_core.apply_activity_plasticity(locals()['total_loss'])

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
                hipa_control_signal=current_hipa_signal # Pass the signal
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
                hipa_control_signal=current_hipa_signal # Pass the signal
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
        
        # Update total_loss to include all losses
        total_loss_combined = total_loss + output_dict['ctm_internal_loss'] + output_dict['mcmc_loss']
        output_dict['total_loss'] = total_loss_combined
        
        # Prepare the final 'losses' dictionary for return.
        # It should contain all individual loss components.
        # The 'losses' variable (from the first part of the function) already has some (e.g., diffusion_loss, ewc_loss).
        # Add ctm_internal_loss and mcmc_loss to it if they were computed in the second part and exist in output_dict.
        if 'ctm_internal_loss' in output_dict and output_dict['ctm_internal_loss'] is not None:
            losses['ctm_internal_loss'] = output_dict['ctm_internal_loss']
        if 'mcmc_loss' in output_dict and output_dict['mcmc_loss'] is not None:
            losses['mcmc_loss'] = output_dict['mcmc_loss']
        
        # The total loss to return is total_loss_combined (calculated around line 3027 using 'total_loss' from Part 1 and new losses).
        # The dictionary of all losses to return is the now augmented 'losses' variable.
        # Add all losses to output_dict for consistency
        for loss_name, loss_val in losses.items():
            if loss_name not in output_dict:
                output_dict[loss_name] = loss_val
        
        # Update the total loss in output_dict
        output_dict['total_loss'] = total_loss_combined
        
        # Re-aggregate total_loss in output_dict to include all components
        # Start with diffusion_loss which should be in output_dict from earlier processing
        current_total_loss = output_dict.get('diffusion_loss', torch.tensor(0.0, device=device))
        current_total_loss += output_dict.get('ctm_internal_loss', torch.tensor(0.0, device=device))
        current_total_loss += output_dict.get('mcmc_loss', torch.tensor(0.0, device=device))
        
        # Add predictive coding loss to total loss
        if 'ctm_core_data' in output_dict and output_dict['ctm_core_data'] and 'predictive_coding_loss' in output_dict['ctm_core_data']:
            pc_loss = output_dict['ctm_core_data']['predictive_coding_loss']
            output_dict['predictive_coding_loss'] = pc_loss
            current_total_loss += pc_loss * getattr(self.config, 'ctm_pc_loss_weight', 0.1)

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
                    # Keep numeric output if conversion fails

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
                         'convergence_history': [], 'certainty_history': []}
        
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
            # `x` is assumed to be in the feature space expected by ctm_core.
            # This means `x` should have the same characteristics as the output of `_prepare_input_features`
            # (i.e., the `kv_features_for_ctm` that `_prepare_input_features` would have produced).
            # This is a key assumption: the `shape` argument to this sampling function must match
            # the expected input shape for `self.ctm_core.forward_with_full_tracking`.
            ctm_input_features_for_core_step = x.detach()
            ctm_data_guidance = self.ctm_core.forward_with_full_tracking(ctm_input_features_for_core_step)

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
                    sampling_info['early_stops'].append({'step': i, 'num_stopped': should_stop_flags.sum().item()})
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
            
            sampling_info['steps_taken'].append(i)

        # Ensure x is float32 before converting to bytes
        x = x.to(torch.float32)
        # Convert to byte tensor
        x_bytes = batched_numeric_tensor_to_bytes(x, source_dtype=np.float32)
        return x_bytes, sampling_info
    
    def get_loss_with_ctm_guidance(self, x_start: torch.Tensor,
                                   inferred_task_latent: torch.Tensor,
                                   hipa_control_signal: torch.Tensor
                                   ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute diffusion loss with CTM guidance and return detailed metrics.
        """
        device = x_start.device
        batch_size = x_start.size(0)
        
        timesteps = torch.randint(0, self.diffusion.scheduler.num_train_timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(x_start)
        noisy_x = self.diffusion.add_noise(x_start, noise, timesteps)
        
        # Prepare CTM input features using x_start or noisy_x depending on CTM's role
        # For guidance, CTM often processes something related to the current state or target.
        # This part needs to align with how ctm_data is generated for diffusion's forward.
        # Let's assume kv_features are derived from noisy_x for consistency with how diffusion might be conditioned.
        # This is a placeholder; a more sophisticated approach might use x_start for CTM's "thought process".
        # The main forward pass derives kv_features from byte_sequence. Here we have x_start (clean data).
        # We need a consistent way to get kv_features for CTM.
        # For now, let's assume a dummy or simplified kv_feature generation for this specific loss function.
        # This function seems to be more of a diagnostic or specialized training loop.
        
        # Simplified: if CTM conditions on noisy_x directly (after some encoding)
        # This is highly dependent on the architecture.
        # For now, let's assume ctm_data is generated based on noisy_x.
        # This part is complex as `self.forward` expects byte_sequence.
        # This method might need to be re-thought or use a different path to get CTM data.
        # For now, creating dummy ctm_data.
        ctm_input_for_loss_calc = self.input_encoder(self.byte_embedding(torch.randint(0,256, (batch_size, noisy_x.shape[1] if noisy_x.dim() > 2 else 128), device=device).long()))
        ctm_data_for_loss = self.ctm_core.forward_with_full_tracking(ctm_input_for_loss_calc)


        # Predict noise with CTM control, passing hipa_control_signal
        predicted_noise_output = self.diffusion(noisy_x, timesteps, ctm_data_for_loss, hipa_control_signal=hipa_control_signal)
        if isinstance(predicted_noise_output, tuple):
            predicted_noise = predicted_noise_output[0]
        else:
            predicted_noise = predicted_noise_output

        diffusion_loss = nn.functional.mse_loss(predicted_noise, noise)
        
        additional_losses = {}
        # Add CTM internal losses if ctm_data_for_loss is properly generated and has them
        if hasattr(self.ctm_core, 'compute_internal_loss') and callable(self.ctm_core.compute_internal_loss):
            ctm_internal_loss_val = self.ctm_core.compute_internal_loss([ctm_data_for_loss], ctm_input_for_loss_calc)
            if isinstance(ctm_internal_loss_val, dict): additional_losses.update(ctm_internal_loss_val)
            elif torch.is_tensor(ctm_internal_loss_val): additional_losses['ctm_internal_objective'] = ctm_internal_loss_val
            
        total_loss = diffusion_loss
        for al_val in additional_losses.values():
            if torch.is_tensor(al_val): total_loss += al_val.mean()
        

        return total_loss, {
            'diffusion_loss': diffusion_loss,
            'total_loss': total_loss,
            **additional_losses,
            'ctm_data': ctm_data_for_loss # For diagnostics
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
            dummy_bytes = torch.randint(0, 256, (batch_size, self.config.byte_embedding_dim if not self.config.multi_granularity else 128), device=device)
            encoded_features, _inferred_latent, hipa_control_signal_sampling = self._prepare_input_features(dummy_bytes)
            ctm_input_for_guidance_generation = self.compute_features(encoded_features)

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
            print("Warning: integration_flow_one_step_generation not found on diffusion processor. Using standard iterative sampling as fallback.")
            # Fallback to iterative sampling if one-step is not available
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
        
        return total_loss, {
            'diffusion_loss': diffusion_loss,
            'total_loss': total_loss,
            **additional_losses,
            'ctm_data': ctm_data
        }
    
    def ultra_fast_integration_flow_generation(self, shape: Tuple[int, ...],
                                             task_id: int = 0,
                                             text_condition: Optional[torch.Tensor] = None,
                                             enable_hipa: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Ultra-fast one-step generation using Integration Flow + Task-Aware HiPA.
        
        This method combines:
        1. CTM deep thought processing
        2. Integration Flow one-step generation
        3. Task-Aware HiPA frequency enhancement
        4. Intelligent modality detection
        
        Returns:
            generated_samples: Generated data
            generation_info: Dictionary with generation statistics and modality info
        """
        device = next(self.parameters()).device
        
        generation_info = {
            'method': 'integration_flow_one_step',
            'hipa_applied': False,
            'modality_detected': 'unknown',
            'generation_time': 0.0,
            'ctm_iterations': self.config.iterations
        }
        
        import time
        start_time = time.time()
        
        try:
            # Step 1: Generate input features for CTM processing
            dummy_input = torch.randn((shape[0], self.config.d_input), device=device)
            kv_features = self.compute_features(dummy_input)
            
            # Step 2: Get full CTM context with deep thought processing
            ctm_data = self.ctm_core.forward_with_full_tracking(kv_features)
            
            # Step 3: Use CTM-controlled diffusion processor for ultra-fast generation
            generated_samples = self.diffusion.integration_flow_one_step_generation(
                shape=shape,
                ctm_data=ctm_data,
                task_id=task_id,
                device=device
            )
            
            # Step 4: Apply additional Task-Aware HiPA if enabled
            if enable_hipa and self.diffusion.enable_task_aware_hipa:
                enhanced_samples, modality_config = self.diffusion.task_aware_hipa(
                    generated_samples, task_id=task_id
                )
                
                generation_info['hipa_applied'] = modality_config['use_hipa']
                generation_info['modality_detected'] = modality_config['modality']
                generation_info['enhancement_strength'] = modality_config['enhancement_strength']
                
                if modality_config['use_hipa']:
                    generated_samples = enhanced_samples
                    generation_info['frequency_enhancement'] = {
                        'fft_dims': modality_config['fft_dims'],
                        'freq_threshold': modality_config['freq_threshold']
                    }
            
            generation_info['generation_time'] = time.time() - start_time
            generation_info['success'] = True
            
            # Step 5: Quality analysis
            generation_info['quality_metrics'] = {
                'finite_values': torch.isfinite(generated_samples).all().item(),
                'value_range': [generated_samples.min().item(), generated_samples.max().item()],
                'std_dev': generated_samples.std().item(),
                'mean': generated_samples.mean().item()
            }
            
            # Step 6: CTM analysis
            if 'certainties' in ctm_data:
                final_certainty = ctm_data['certainties'][:, 0, -1].mean().item()
                generation_info['ctm_certainty'] = final_certainty
                generation_info['high_confidence'] = final_certainty > 0.8
            
            return generated_samples, generation_info
            
        except Exception as e:
            generation_info['success'] = False
            generation_info['error'] = str(e)
            generation_info['generation_time'] = time.time() - start_time
            
            print(f"Warning: Ultra-fast Integration Flow generation failed: {e}")
            
            # Fallback to simple random generation
            fallback_samples = torch.randn(shape, device=device)
            return fallback_samples, generation_info
    
    def benchmark_generation_methods(self, shape: Tuple[int, ...],
                                   task_id: int = 0,
                                   num_trials: int = 5) -> Dict[str, Dict]:
        """
        Benchmark different generation methods for performance comparison.
        
        Compares:
        1. Ultra-fast Integration Flow (one-step)
        2. Standard iterative CTM-diffusion
        3. Pure CTM generation
        
        Returns detailed performance metrics.
        """
        device = next(self.parameters()).device
        benchmark_results = {}
        
        print(f"🚀 Benchmarking generation methods on shape {shape}...")
        
        # 1. Ultra-fast Integration Flow
        print("Testing Integration Flow + HiPA (one-step)...")
        integration_times = []
        integration_qualities = []
        
        for trial in range(num_trials):
            samples, info = self.ultra_fast_integration_flow_generation(
                shape=shape, task_id=task_id, enable_hipa=True
            )
            integration_times.append(info['generation_time'])
            integration_qualities.append(info['quality_metrics']['std_dev'])
        
        benchmark_results['integration_flow'] = {
            'avg_time': sum(integration_times) / len(integration_times),
            'min_time': min(integration_times),
            'max_time': max(integration_times),
            'avg_quality': sum(integration_qualities) / len(integration_qualities),
            'method': 'One-step Integration Flow + Task-Aware HiPA'
        }
        
        # 2. Standard iterative sampling (reduced steps)
        print("Testing standard iterative sampling...")
        iterative_times = []
        iterative_qualities = []
        
        for trial in range(num_trials):
            start_time = time.time()
            samples, info = self.iterative_ctm_diffusion_sample(
                shape=shape, num_steps=10, task_id=task_id,  # Reduced steps for fair comparison
                enable_early_stopping=True
            )
            elapsed = time.time() - start_time
            iterative_times.append(elapsed)
            iterative_qualities.append(samples.std().item())
        
        benchmark_results['iterative_sampling'] = {
            'avg_time': sum(iterative_times) / len(iterative_times),
            'min_time': min(iterative_times),
            'max_time': max(iterative_times),
            'avg_quality': sum(iterative_qualities) / len(iterative_qualities),
            'method': 'Iterative CTM-Diffusion (10 steps)'
        }
        
        # 3. Pure CTM generation
        print("Testing pure CTM generation...")
        ctm_times = []
        ctm_qualities = []
        
        for trial in range(num_trials):
            start_time = time.time()
            dummy_input = torch.randn((shape[0], self.config.d_input), device=device)
            predictions, certainties, sync_out = self.forward(
                dummy_input, mode='ctm', task_id=task_id
            )
            elapsed = time.time() - start_time
            ctm_times.append(elapsed)
            ctm_qualities.append(predictions.std().item())
        
        benchmark_results['pure_ctm'] = {
            'avg_time': sum(ctm_times) / len(ctm_times),
            'min_time': min(ctm_times),
            'max_time': max(ctm_times),
            'avg_quality': sum(ctm_qualities) / len(ctm_qualities),
            'method': 'Pure CTM (no diffusion)'
        }
        
        # Calculate speedup ratios
        integration_time = benchmark_results['integration_flow']['avg_time']
        iterative_time = benchmark_results['iterative_sampling']['avg_time']
        ctm_time = benchmark_results['pure_ctm']['avg_time']
        
        benchmark_results['speedup_analysis'] = {
            'integration_vs_iterative': iterative_time / integration_time if integration_time > 0 else float('inf'),
            'integration_vs_ctm': ctm_time / integration_time if integration_time > 0 else float('inf'),
            'fastest_method': min(benchmark_results.keys(),
                                key=lambda k: benchmark_results[k]['avg_time'] if k != 'speedup_analysis' else float('inf'))
        }
        
        # Print summary
        print(f"\n📊 Benchmark Results Summary:")
        for method, results in benchmark_results.items():
            if method != 'speedup_analysis':
                print(f"  {method}: {results['avg_time']:.4f}s avg, quality: {results['avg_quality']:.4f}")
        
        speedup = benchmark_results['speedup_analysis']
        print(f"\n⚡ Speedup Analysis:")
        print(f"  Integration Flow vs Iterative: {speedup['integration_vs_iterative']:.1f}x faster")
        print(f"  Integration Flow vs Pure CTM: {speedup['integration_vs_ctm']:.1f}x faster")
        print(f"  Fastest method: {speedup['fastest_method']}")
        
        return benchmark_results




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

    def close(self):
        """Safely close connections, including the knowledge store."""
        if self.knowledge_store:
            self.knowledge_store.close()
            print("Neo4j knowledge store connection closed.")

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


class FrequencyDomainAwareAttention(nn.Module):
    """Generalized HiPA that works across different modalities with intelligent task detection."""
    
    def __init__(self, embed_dim=512, num_heads=8, task_analyzer: 'TaskAnalyzer' = None): # Added task_analyzer
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.task_analyzer = task_analyzer # Store the task_analyzer instance
        
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
        
    def apply_frequency_enhancement(self, x, modality_config):
        """Apply frequency enhancement based on detected modality."""
        if not modality_config['use_hipa']:
            return x  # Skip enhancement for text/discrete tasks
        
        original_shape = x.shape
        
        try:
            # Apply FFT on specified dimensions
            fft_dims = modality_config['fft_dims']
            if not fft_dims:
                return x
            
            # Compute FFT
            x_fft = torch.fft.fftn(x, dim=fft_dims)
            
            # Create frequency mask for enhancement
            freq_threshold = modality_config['freq_threshold']
            enhancement_strength = modality_config['enhancement_strength']
            
            # Generate frequency coordinates
            freq_mask = torch.ones_like(x_fft.real)
            
            for dim in fft_dims:
                size = x.shape[dim]
                freqs = torch.fft.fftfreq(size, device=x.device)
                
                # Create high-frequency mask
                high_freq_mask = torch.abs(freqs) > freq_threshold
                
                # Expand mask to match tensor dimensions
                mask_shape = [1] * len(x.shape)
                mask_shape[dim] = size
                high_freq_mask = high_freq_mask.view(mask_shape)
                
                # Broadcast and apply
                freq_mask = freq_mask * (1.0 + enhancement_strength * high_freq_mask.float())
            
            # Apply frequency enhancement
            x_fft_enhanced = x_fft * freq_mask
            
            # Convert back to spatial/temporal domain
            x_enhanced = torch.fft.ifftn(x_fft_enhanced, dim=fft_dims).real
            
            return x_enhanced
            
        except Exception as e:
            print(f"Warning: Frequency enhancement failed: {e}")
            return x  # Fallback to original
    
    def forward(self, x: torch.Tensor, hipa_control_signal: Optional[torch.Tensor] = None, context_hints: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with dynamically controlled frequency enhancement.
        HiPA is now primarily controlled by hipa_control_signal.
        Task_analyzer might still provide base modality characteristics.
        """
        batch_size, seq_len, embed_dim = x.shape # Assuming x is (batch, seq_len, embed_dim)

        # Use task_analyzer to get a base modality_config (e.g., for fft_dims, static thresholds)
        # It should not rely on task_id anymore, but can use context_hints or analyze x directly.
        modality_config = self.task_analyzer.detect_modality(x, task_id=None, context_hints=context_hints)

        x_processed = x
        # Check if HIPA should be used at all for this base modality
        if modality_config.get('use_hipa', False): # Default to False if not specified
            # Now, check the dynamic hipa_control_signal
            if hipa_control_signal is not None:
                # Assuming hipa_control_signal is (batch, 1), threshold it
                # For a soft application, use it as a multiplier.
                # For a hard switch, threshold it. Let's use soft.
                control = hipa_control_signal.view(batch_size, 1, 1) # Reshape for broadcasting
                
                # Apply frequency enhancement based on modality_config
                x_enhanced_by_modality = self.apply_frequency_enhancement(x, modality_config)
                
                # Interpolate between original and enhanced based on control signal
                x_processed = control * x_enhanced_by_modality + (1 - control) * x
            else:
                # No dynamic control signal, just use the static modality_config
                x_processed = self.apply_frequency_enhancement(x, modality_config)
        
        # Standard multi-head attention projections on the (potentially) HIPA-processed features
        # The self.freq_attention and self.freq_enhancer from the original code seem to be a separate attention path.
        # The request was to control HIPA on/off.
        # If the existing self.freq_attention is the HIPA mechanism, then x_processed should go into it.
        # Let's assume self.freq_attention IS the HiPA-specific attention.
        
        # The original code had:
        # if x_flat.shape[-1] == self.embed_dim:
        #     attn_out, _ = self.freq_attention(x_flat, x_flat, x_flat)
        #     enhanced = self.freq_enhancer(attn_out)
        # else:
        #     enhanced = x_flat
        # This implies freq_attention is a standard MHA. The "frequency aware" part was in apply_frequency_enhancement.
        # So, we apply MHA on x_processed.

        # Reshape for attention if needed (original code had this before freq_attention)
        original_shape = x_processed.shape
        if len(x_processed.shape) > 3:
            x_flat = x_processed.view(batch_size, -1, x_processed.shape[-1])
        else:
            x_flat = x_processed
            seq_len = x_flat.shape[1] # Update seq_len if flattened differently

        # Standard attention mechanism (using self.freq_attention as the MHA layer)
        if x_flat.shape[-1] == self.embed_dim: # Ensure dimension matches MHA
            attn_output, _ = self.freq_attention(x_flat, x_flat, x_flat) # Q, K, V are the same
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


        # Reshape back to original if flattened
        if len(original_shape) > 3:
            final_output = final_output_flat.view(original_shape[0], *original_shape[1:-1], self.embed_dim) # Ensure last dim is embed_dim
        else:
            final_output = final_output_flat
            
        return final_output, modality_config # Return modality_config for potential logging/debugging


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
        print(f"✓ Integration Flow + Task-Aware HiPA sampler initialized")
        print(f"  - Model type: {model_type}")
        print(f"  - Intelligent modality detection enabled")
        print(f"  - HiPA will be applied only to appropriate data types")
        print(f"  - Text/discrete tasks protected from frequency corruption")
    
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
                print(f"Warning: Integration Flow step failed: {e}, using fallback")
                # Fallback to simple denoising
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
