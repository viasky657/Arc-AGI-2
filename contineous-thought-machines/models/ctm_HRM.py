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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math
import numpy as np

from .modules import SynapseUNET, SuperLinear, Squeeze
from .ctm_Diffusion_NEWNEW import (
    WINAAttention, WINAEnhancedMLP, EnhancedCTMConfig, OriginalCTMCore, 
    SubquadraticAttention, BinarySparseAttention
)
from .utils import compute_normalized_entropy

class HRM_L_Module(nn.Module):
    """The Low-Level, fast-updating CTM-based recurrent module for the HR-CTM."""
    def __init__(self, config: EnhancedCTMConfig, parent_ctm: 'HierarchicalCTM'):
        super().__init__()
        self.config = config
        self.d_model = config.ctm_d_model
        self.d_input = config.ctm_input_dim
        
        # Inherit synapse and NLM models from parent HierarchicalCTM
        # to ensure they are registered correctly under the main model.
        self.synapses = parent_ctm.synapses
        self.trace_processor = parent_ctm.trace_processor
        
        # Attention to combine external input (x) and high-level context (zH)
        self.attention = SubquadraticAttention(
            embed_dim=self.d_input,
            num_heads=config.ctm_heads,
            qkv_bias=config.attention_qkv_bias,
            attn_drop=config.ctm_dropout,
            proj_drop=config.ctm_dropout
        )
        # Projector for the query, derived from the low-level sync state
        self.q_proj = nn.Linear(parent_ctm.synch_representation_size_action, self.d_input)

    def forward(self, 
                activated_zL: torch.Tensor, 
                zL_trace: torch.Tensor, 
                zH: torch.Tensor, 
                x_context: torch.Tensor,
                sync_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # 4. Update state trace (memory for NLMs)
        next_zL_trace = torch.cat((zL_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

        # 5. Apply Neuron-Level Models (NLMs) to get next post-activation state
        next_activated_zL = self.trace_processor(next_zL_trace)
        
        return next_activated_zL, next_zL_trace


class HRM_H_Module(nn.Module):
    """The High-Level, slow-updating recurrent module for the HR-CTM."""
    def __init__(self, config: EnhancedCTMConfig):
        super().__init__()
        self.config = config
        # This module integrates the result from the L-module (zL) into its own state (zH).
        self.attn = WINAAttention(d_model=config.d_model, n_heads=config.n_heads, dropout=config.dropout)
        self.mlp = WINAEnhancedMLP(d_model=config.d_model, d_ff=config.d_model * 4, dropout=config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        # Project zL to match d_model for attention
        self.zl_proj = nn.Linear(config.d_model, config.d_model) # Assuming zL has d_model

    def forward(self, zH: torch.Tensor, zL: torch.Tensor) -> torch.Tensor:
        """
        Args:
            zH (torch.Tensor): Current high-level state.
            zL (torch.Tensor): Final low-level state from the L-cycle.
        Returns:
            torch.Tensor: Next high-level state.
        """
         # The query is the current high-level state
        q = zH
        # The key/value is the information from the completed low-level cycle
        kv = self.zl_proj(zL)
        # Attention step
        attn_output = self.attn(q, kv, kv)
        zH = self.norm1(zH + attn_output)
        # MLP step
        mlp_output = self.mlp(zH)
        zH = self.norm2(zH + mlp_output)
        return zH

class HierarchicalCTM(OriginalCTMCore):
    """
    The main Hierarchical Reasoning CTM model.
    Inherits from OriginalCTMCore to reuse helper methods for initialization."
    """
    def __init__(self, config: EnhancedCTMConfig):
        # We call nn.Module's init directly to avoid OriginalCTMCore's full init,
        # as we are building a different structure.
        super(OriginalCTMCore, self).__init__()
        self.config = config
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

    def forward_with_full_tracking(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        The main forward pass implementing the hierarchical reasoning process.
        This method will replace the original CTM's iterative loop.
        """
        b, s, _ = x.shape
        device = x.device

        # 1. Project input 
        x_context = self.input_encoder(x)
        
        # 2. Initialize states
        activated_zL = self.start_activated_zL.unsqueeze(0).expand(b, -1)
        zL_trace = self.start_trace_zL.unsqueeze(0).expand(b, -1, -1)
        zH = self.start_zH.unsqueeze(0).expand(b, -1)
        
        # 3. Initialize sync recurrent values
        decay_alpha_action, decay_beta_action = None, None
        decay_alpha_out, decay_beta_out = None, None
        r_action = torch.exp(-self.decay_params_action).unsqueeze(0).expand(b, -1)
        r_out = torch.exp(-self.decay_params_out).unsqueeze(0).expand(b, -1)

        # Store history of high-level states for final representation
        zH_history = []

        # 4. Hierarchical recurrent loop
        for n in range(self.config.hrm_high_level_cycles):
            # The L-module's own synchronisation state is reset/recalculated each high-level cycle
            decay_alpha_action, decay_beta_action = None, None

            for t in range(self.config.hrm_low_level_timesteps):
                # Compute L-module's action synchronisation for its attention query
                sync_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(
                    activated_zL, decay_alpha_action, decay_beta_action, r_action, 'action'
                )
                
                # Run one step of the L-module
                activated_zL, zL_trace = self.l_module(
                    activated_zL, zL_trace, zH, x_context, sync_action
                )
            
            # End of low-level cycle, update high-level state using the final L-state
            zH = self.h_module(zH, activated_zL)
            zH_history.append(zH)

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

        return {
            'predictions': predictions.unsqueeze(-1),
            'certainties': certainties.unsqueeze(-1),
            'abstained': abstain_mask.unsqueeze(-1),
            'final_sync_out': synchronisation_out,
            'activated_states': zH_history,
        }
