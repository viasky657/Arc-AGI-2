"""
# ======================================================================================
#
#               The Unified CTM Denoising Model Architecture
#
# ======================================================================================
#
# Author: Roo
# Date: 2025-07-26
#
# --- Architecture Philosophy ---
# This model implements the most ambitious fusion of the Continuous Thought Machine (CTM)
# and a diffusion model. Instead of treating the CTM as a separate "controller" that
# guides a diffusion "actuator," this architecture refactors the CTM to BE the
# denoising network itself.
#
# The core insight is that the iterative, recurrent "thinking" process inherent to the
# CTM is functionally equivalent to the iterative refinement process of denoising
# in a diffusion model. Each "thought" tick of the CTM becomes a step in refining
# the prediction of the noise that was added to the clean data. This has been updated 
# to predict velocity `v` for a rectified flow model.
#
# --- Key Architectural Changes ---
#
# 1.  **Unified Model Class:** A new `UnifiedCTMDenoisingModel` class is created. This
#     class is fundamentally based on the `HierarchicalCTM` but is repurposed for its new
#     role. It no longer contains a separate diffusion processor instance.
#
# 2.  **Modified Forward Pass:** The `forward` method's signature is fundamentally
#     changed. It now accepts `(noisy_input, timestep)` as its primary arguments,
#     mirroring the standard API for a diffusion model's U-Net.
#
# 3.  **Timestep Injection:** The timestep embedding is a crucial piece of conditioning
#     information that tells the model at what point in the diffusion process it is
#     operating. This embedding is injected directly into every step of the CTM's
#     internal recurrent loop. This ensures the "thought" process is always aware 
#     of the denoising timeline.
#
# 4.  **Output as Velocity Prediction:** The model's final output is no longer a high-level
#     "thought vector." Instead, the final synchronization vector produced by the CTM
#     is projected to have the same dimensionality as the input data. This projected
#     output represents the model's prediction of the velocity `v` required to travel
#     from the noisy data `x_t` to the clean data `x_0`.
#
# 5.  **Simplified Loss:** The training objective becomes the standard diffusion loss:
#     Mean Squared Error between the predicted velocity and the actual velocity.
#
# --- How It Works ---
#
# | Step    | Process                                                              |
# |---------|----------------------------------------------------------------------|
# | Input   | A noisy data sample `x_t` and a timestep `t`.                        |
# | Embed   | A timestep embedding is created from `t`.                            |
# | Iterate | The CTM's recurrent loop begins. At each "thought" iteration:        |
# |         |   a. The model's internal state is updated.                          |
# |         |   b. The timestep embedding is injected into the update calculation. |
# | Output  | After all iterations, the final CTM state is used to predict `v`.    |
# | Loss    | The model is trained to minimize `MSE(predicted_v, actual_v)`.   |
#
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .ctm_components import (
    EnhancedCTMConfig,
    HierarchicalCTM,
    WINAAttention
)
from diffusers import DPMSolverMultistepScheduler
import numpy as np


class UnifiedCTMDenoisingModel(HierarchicalCTM):
    """
    This model reframes the HierarchicalCTM as the core denoising network
    in a rectified flow (velocity prediction) diffusion process.

    It inherits the structure and internal dynamics of the HierarchicalCTM
    but modifies its forward pass to accept a noisy input and a timestep,
    predicting the velocity required to denoise the input.
    """

    def __init__(self, config: EnhancedCTMConfig):
        # Call the parent constructor to set up all the CTM components
        super().__init__(config)

        # 1. Timestep Embedding Network
        self.time_embedding = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # 2. Input Projection for Noisy Data
        # Projects the noisy input to the dimension expected by the CTM's attention mechanism.
        self.noisy_input_projection = nn.Linear(config.unet_input_feature_dim, config.ctm_input_dim)

        # ADDED: WINA Cross-Attention for conditioning
        self.conditioning_cross_attention = WINAAttention(
            d_model=config.ctm_input_dim,
            n_heads=config.n_heads,
            config=config,
            dropout=config.dropout
        )

        # 3. Final Output Projection
        # Projects the CTM's final thought vector to the velocity vector dimension.
        self.final_velocity_projection = nn.Linear(
            self.synch_representation_size_out, config.unet_input_feature_dim
        )

        # 4. Rectified Flow Scheduler
        self.scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_timesteps,
            beta_start=config.diffusion_beta_start,
            beta_end=config.diffusion_beta_end,
            beta_schedule="linear"
        )

    def forward(
        self,
        noisy_input: torch.Tensor,
        timestep: torch.Tensor,
        conditioning_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        The main forward pass for the unified denoising model.
        """
        # 1. Embed the timestep
        time_emb = self.time_embedding(timestep.float().unsqueeze(-1))

        # 2. Project the noisy input and add sequence dimension for CTM 'x' input
        projected_input = self.noisy_input_projection(noisy_input).unsqueeze(1)

        # 3. Apply Cross-Attention Conditioning
        if conditioning_features is not None:
            # Use the projected noisy input as the query and conditioning features as key/value
            projected_input = self.conditioning_cross_attention(
                query=projected_input,
                key=conditioning_features,
                value=conditioning_features
            )

        # 4. Run the Hierarchical CTM's reasoning process, conditioned on time
        ctm_output_data = self.hrm_forward_with_full_tracking(
            x=projected_input,
            time_embedding=time_emb
        )

        # 4. Extract the final "thought vector" and project to velocity
        final_sync_out = ctm_output_data['final_sync_out']
        predicted_velocity = self.final_velocity_projection(final_sync_out)

        return predicted_velocity

    def hrm_forward_with_full_tracking(self,
       x: torch.Tensor,
       time_embedding: torch.Tensor,
       confidence_level: str = 'medium',
       thought_guidance: bool = True,
       voice1_id: Optional[torch.Tensor] = None,
       voice2_id: Optional[torch.Tensor] = None,
       blend_degree: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

       b, s, _ = x.shape
       device = x.device
       self.consciousness_controller.wake_up(0)

       x_context = self.input_encoder(x)
       activated_zL = self.start_activated_zL.unsqueeze(0).expand(b, -1)
       zL_trace = self.start_trace_zL.unsqueeze(0).expand(b, -1, -1)
       zH = self.start_zH.unsqueeze(0).expand(b, -1) + time_embedding
        
       decay_alpha_action, decay_beta_action = None, None
       decay_alpha_out, decay_beta_out = None, None
       r_action = torch.exp(-self.decay_params_action).unsqueeze(0).expand(b, -1)
       r_out = torch.exp(-self.decay_params_out).unsqueeze(0).expand(b, -1)
            
       zH_history = []

       for n in range(self.config.hrm_high_level_cycles):
            decay_alpha_action, decay_beta_action = None, None
            prev_zL = activated_zL.clone()
            for t in range(self.config.hrm_low_level_timesteps):
                sync_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(
                    activated_zL, decay_alpha_action, decay_beta_action, r_action, 'action'
                )
                if self.basal_ganglia:
                    action_candidates = [sync_action, sync_action * 0.5, sync_action * 1.5]
                    sync_action = self.basal_ganglia.select_action(action_candidates, activated_zL, x_context.mean(dim=1))
                
                modified_zH = zH + time_embedding
                
                activated_zL, zL_trace = self.l_module(
                    activated_zL, zL_trace, modified_zH, x_context, sync_action, confidence_level=confidence_level
                )
                if self.config.enable_neuromodulators and hasattr(self, 'neuromodulators') and hasattr(self, 'mod_fusion'):
                    mod_outputs = [mod(activated_zL) for mod in self.neuromodulators.values()]
                    concatenated_mods = torch.cat(mod_outputs, dim=-1)
                    fused_mod = self.mod_fusion(concatenated_mods)
                    activated_zL = activated_zL * fused_mod
                activated_zL = self.working_memory.update(activated_zL)
                delta = torch.norm(activated_zL - prev_zL)
                if delta < 1e-3: break
                prev_zL = activated_zL.clone()
            
            surprise = F.mse_loss(activated_zL, zH)
            if surprise > self.config.ltm_surprise_threshold:
                self.ltm.add_to_memory(zH.squeeze(0), surprise)

            retrieved_memory, _ = self.ltm.retrieve(activated_zL, top_k=1)
            fused_input = self.fusion_proj(torch.cat([activated_zL, retrieved_memory.squeeze(1)], dim=-1))
            
            modified_zH_h = zH + time_embedding
            
            zH, _, _, _, _ = self.h_module(modified_zH_h, fused_input, retrieved_memory, thought_guidance=thought_guidance, confidence_level=confidence_level)

            zH_history.append(zH)
       # Final output from the last H-state
       final_zH_trace = torch.stack(zH_history, dim=-1)
       final_sync_out, _, _ = self.compute_synchronisation(
           final_zH_trace[:,:, -1], None, None, r_out, 'out'
       )

       return { 'final_sync_out': final_sync_out }


    # Add sampling and training helper methods
    def add_noise(self, x_start: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to a clean sample for rectified flow.
        x_t = t * x_1 + (1 - t) * x_0
        """
        t = timesteps.float() / (self.config.diffusion_timesteps - 1)
        while len(t.shape) < len(x_start.shape):
            t = t.unsqueeze(-1)
        x_t = t * noise + (1 - t) * x_start
        return x_t

    def get_velocity(self, x_start: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Computes the target velocity for rectified flow.
        v = x_1 - x_0
        """
        return noise - x_start

    def sample(
        self,
        shape: Tuple[int, ...],
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        conditioning_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generates a sample using the unified denoising model.
        """
        device = self.final_velocity_projection.weight.device
        x = torch.randn(shape, device=device, generator=generator)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        for t in timesteps:
            timestep_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict velocity
            velocity = self.forward(x, timestep_batch, conditioning_features=conditioning_features)
            
            # Denoise one step
            x = self.scheduler.step(velocity, t, x).prev_sample
            
        return x
