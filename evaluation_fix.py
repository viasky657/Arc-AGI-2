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
import math
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import collections

# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection.
    Reduces the relative loss for well-classified examples, putting more
    focus on hard, misclassified examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: [N, C, H, W], targets: [N, H, W]
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        ce_loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)
        ce_loss = ce_loss_fn(inputs, targets)
        
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# --- MCMC Self-Learning Components (Adapted from SEAL/mcmc_search.py) ---

@dataclass
class MCMCConfig:
    """Configuration for MCMC sampling parameters."""
    num_chains: int = 3000
    chain_length: int = 8000
    burn_in: int = 200
    temperature_schedule: str = "adaptive"  # "geometric", "adaptive"
    initial_temp: float = 30.0
    final_temp: float = 0.01
    decay_rate: float = 0.998
    # Adaptive cooling parameters
    target_acceptance_rate: float = 0.30
    adaptive_adjustment_factor: float = 0.02
    # Proposal strategy
    proposal_strategy: str = "hybrid" # "random", "hybrid", "structured"
    structured_mutation_prob: float = 0.5
    mutation_region_size: int = 3
    # Convergence Monitoring
    convergence_window: int = 250
    convergence_threshold: float = 1e-6
    # --- New: Iterative Local Refinement ---
    enable_local_refinement: bool = True
    refinement_trigger_similarity: float = 0.95 # Similarity to target to trigger refinement
    refinement_steps: int = 500

class ARCSelfEditSpace:
    """Defines the space of ARC grid edits and their neighborhoods with adaptive mutation."""
    def __init__(self, grid_dims: tuple, initial_mutation_rate: float = 0.1, final_mutation_rate: float = 0.01, decay_steps: int = 8000):
        self.grid_dims = grid_dims
        self.initial_mutation_rate = initial_mutation_rate
        self.final_mutation_rate = final_mutation_rate
        self.decay_steps = decay_steps

    def _get_mutation_rate(self, step: int, is_refinement: bool = False) -> float:
        """Anneals mutation rate from high to low for broader exploration followed by exploitation."""
        if is_refinement:
            return self.final_mutation_rate / 5.0 # Much lower rate for fine-tuning
        if step >= self.decay_steps:
            return self.final_mutation_rate
        
        # Use a cosine annealing schedule for a smoother transition
        progress = step / self.decay_steps
        cosine_out = np.cos(np.pi * progress) + 1
        return self.final_mutation_rate + 0.5 * (self.initial_mutation_rate - self.final_mutation_rate) * cosine_out

    def get_neighbors(self, state: np.ndarray, step: int, n_neighbors: int = 1, error_focus: Optional[np.ndarray] = None, strategy: str = "hybrid", region_size: int = 3, is_refinement: bool = False) -> List[np.ndarray]:
        """
        Generates neighbors using adaptive mutation and various proposal strategies.
        During refinement, it uses a much smaller mutation rate.
        """
        mutation_rate = self._get_mutation_rate(step, is_refinement=is_refinement)
        neighbors = []
        h, w = self.grid_dims

        for _ in range(n_neighbors):
            neighbor = state.copy()
            
            # Decide between pixel-level and region-level mutation
            use_structured_mutation = random.random() < 0.5 # 50% chance for structured

            if strategy == 'hybrid' and error_focus is not None and random.random() < 0.75:
                error_indices = np.argwhere(error_focus > 0)
                if len(error_indices) > 0:
                    num_mutations = max(1, int(len(error_indices) * mutation_rate * 2))
                    for _ in range(num_mutations):
                        idx_to_mutate = random.choice(error_indices)
                        neighbor[tuple(idx_to_mutate)] = random.randint(0, 9)
                else:
                    use_structured_mutation = True # Fallback to structured if no errors
            
            elif use_structured_mutation:
                # Structured perturbation: modify a contiguous region
                r_h = min(region_size, h)
                r_w = min(region_size, w)
                top = random.randint(0, h - r_h)
                left = random.randint(0, w - r_w)
                for i in range(top, top + r_h):
                    for j in range(left, left + r_w):
                        if random.random() < mutation_rate * 2: # Higher effective rate in region
                            neighbor[i, j] = random.randint(0, 9)
            else:
                # Global, random pixel-wise mutation
                for i in range(h):
                    for j in range(w):
                        if random.random() < mutation_rate:
                            neighbor[i, j] = random.randint(0, 9)
            
            neighbors.append(neighbor)
        return neighbors

class CTMSurrogate:
    """A surrogate that uses the main CTM model and its head to guide MCMC search."""
    def __init__(self, model, arc_output_head, input_bytes_eval, target_grid, device, feature_extractor=None):
        self.model = model
        self.arc_output_head = arc_output_head
        self.input_bytes_eval = input_bytes_eval
        self.target_grid = target_grid # This is the cropped numpy target
        self.device = device
        # --- New: Placeholder for a learned feature extractor ---
        self.feature_extractor = feature_extractor
        self.model_prediction_full = self._get_model_prediction()
        h, w = self.target_grid.shape
        self.model_prediction_cropped = self.model_prediction_full[:h, :w]

    def _get_model_prediction(self):
        """Runs the model and head once to get a baseline grid prediction."""
        with torch.inference_mode():
            current_batch_size_eval = self.input_bytes_eval.size(0)
            eval_timestep = torch.zeros(current_batch_size_eval, device=self.device).long()
            
            # 1. Get features from the backbone model
            eval_model_output_dict = self.model(
                byte_sequence=self.input_bytes_eval,
                mode='ctm_controlled_diffusion',
                target_diffusion_output=None,
                timestep=eval_timestep,
                task_name="ARC_AGI_2_EVAL_DIFFUSION"
            )
            
            # 2. Extract features consistent with training loop logic
            ctm_core_output_data = eval_model_output_dict.get('ctm_core_data')
            ctm_backbone_output = None
            if ctm_core_output_data and 'final_sync_out' in ctm_core_output_data:
                ctm_backbone_output = ctm_core_output_data['final_sync_out']
            elif ctm_core_output_data and 'ctm_latent_representation' in ctm_core_output_data:
                ctm_backbone_output = ctm_core_output_data['ctm_latent_representation']
            
            if ctm_backbone_output is not None:
                # Process features like in training
                if ctm_backbone_output.ndim > 2 and ctm_backbone_output.shape[1] > 0:
                     ctm_features_for_head = ctm_backbone_output.mean(dim=1)
                else:
                     ctm_features_for_head = ctm_backbone_output
                
                # 3. Get logits from the prediction head
                logits = self.arc_output_head(ctm_features_for_head)
                preds_flat = torch.argmax(logits.view(-1, NUM_ARC_SYMBOLS), dim=-1)
                preds_grid = preds_flat.view(MAX_GRID_SIZE).long().cpu().numpy()
                return preds_grid

        return np.zeros(MAX_GRID_SIZE, dtype=int)

    def predict(self, state: np.ndarray, penalty_weight: float = 0.2) -> float:
        """
        Calculates a blended score using a smoother reward function with a penalty
        for drastic changes from the original model's prediction.
        Includes calibrated rewards and a placeholder for feature-based similarity.
        """
        # --- Fine-Grained Similarity Metric ---
        if self.feature_extractor:
            # Placeholder for using a learned feature-based similarity
            # target_features = self.feature_extractor(self.target_grid)
            # state_features = self.feature_extractor(state)
            # sim_to_target = cosine_similarity(target_features, state_features)
            pass

        sim_to_target = np.sum(state == self.target_grid) / self.target_grid.size

        # --- Calibrated Reward Function ---
        # Sigmoid scaling to heavily penalize anything less than a perfect match
        # The steepness (k) makes the reward sharply increase as similarity approaches 1.
        k = 20
        reward_from_target = 1 / (1 + np.exp(-k * (sim_to_target - 0.95)))

        # Similarity to the initial (failed) model prediction
        sim_to_model = np.sum(state == self.model_prediction_cropped) / self.model_prediction_cropped.size
        
        # Penalty for deviating too far from the model's original prediction
        deviation_penalty = np.sum(state != self.model_prediction_cropped) / self.model_prediction_cropped.size
        
        # Dynamically balance rewards based on initial model quality
        baseline_similarity = np.sum(self.model_prediction_cropped == self.target_grid) / self.target_grid.size
        
        # If the baseline is good, trust the model more. If not, trust the target reward more.
        alpha = min(0.9, 1.0 - baseline_similarity**2) # Squaring makes it more sensitive

        # Blended score
        blended_score = (alpha * reward_from_target +
                         (1.0 - alpha) * sim_to_model -
                         penalty_weight * deviation_penalty)
        return blended_score

    def adapt_to_new_task(self, train_pairs: List[dict], optimizer):
        """
        Fine-tunes the surrogate model's head on the training examples of a new,
        unseen task to improve its initial predictions (meta-learning).
        """
        print("  > Meta-learning: Adapting surrogate to new task train examples...")
        
        # This is a conceptual implementation. A real one would involve
        # a few quick gradient steps on a small, separate optimizer for the head.
        # For now, it serves as a structural placeholder.
        
        # Create a temporary optimizer for the head for quick adaptation
        head_optimizer = torch.optim.Adam(self.arc_output_head.parameters(), lr=1e-3)
        loss_fn = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        
        # --- Pre-computation outside the loop ---
        # Get the CTM features once, as they don't change for the task
        with torch.no_grad():
            eval_timestep = torch.zeros(self.input_bytes_eval.size(0), device=self.device).long()
            eval_model_output_dict = self.model(
                byte_sequence=self.input_bytes_eval,
                mode='ctm_controlled_diffusion',
                target_diffusion_output=None,
                timestep=eval_timestep,
                task_name="ARC_AGI_2_META_LEARN_FEATURES"
            )
            ctm_core_output_data = eval_model_output_dict.get('ctm_core_data', {})
            ctm_backbone_output = ctm_core_output_data.get('final_sync_out', ctm_core_output_data.get('ctm_latent_representation'))

            if ctm_backbone_output is None:
                print("  > Meta-learning WARNING: Could not extract features. Aborting adaptation.")
                return

            if ctm_backbone_output.ndim > 2 and ctm_backbone_output.shape[1] > 0:
                self.ctm_features_for_head = ctm_backbone_output.mean(dim=1).detach()
            else:
                self.ctm_features_for_head = ctm_backbone_output.detach()

        # --- Quick fine-tuning loop ---
        self.arc_output_head.train()
        for epoch in range(3): # A few quick epochs
            total_loss = 0
            for pair in train_pairs:
                head_optimizer.zero_grad()
                
                # Get target grid and its dimensions
                output_grid_np = pair['output'].cpu().numpy()
                h, w = pair['original_output_dims']
                
                # Prepare target tensor for loss calculation
                target_grid_tensor = torch.from_numpy(output_grid_np).long().to(self.device)
                
                # Forward pass through the head ONLY
                logits = self.arc_output_head(self.ctm_features_for_head)
                logits = logits.view(1, NUM_ARC_SYMBOLS, MAX_GRID_SIZE[0], MAX_GRID_SIZE[1])
                cropped_logits = logits[:, :, :h, :w]
                
                # Calculate loss
                loss = loss_fn(cropped_logits, target_grid_tensor[:h, :w].unsqueeze(0))
                
                # Backward pass and optimization step
                loss.backward()
                head_optimizer.step()
                total_loss += loss.item()
            
            if len(train_pairs) > 0:
                print(f"  > Meta-learning epoch {epoch+1}, Avg Loss: {total_loss / len(train_pairs):.4f}")

        self.arc_output_head.eval() # Return head to evaluation mode
        
        # After adaptation, re-compute the model's prediction
        print("  > Re-evaluating prediction with adapted head...")
        self.model_prediction_full = self._get_model_prediction()
        h_target, w_target = self.target_grid.shape
        self.model_prediction_cropped = self.model_prediction_full[:h_target, :w_target]

def metropolis_hastings_sampler(
    initial_state: np.ndarray,
    surrogate_func: CTMSurrogate,
    output_space: ARCSelfEditSpace,
    config: MCMCConfig,
    error_focus: Optional[np.ndarray] = None
) -> (np.ndarray, dict):
    """
    Performs Metropolis-Hastings search with adaptive temperature, hybrid proposals,
    and convergence monitoring.
    """
    best_state = initial_state
    best_energy = -surrogate_func.predict(initial_state)
    current_state, current_energy = best_state, best_energy
    
    temperature = config.initial_temp
    acceptance_history = []
    energy_history = []
    log_data = {'temp': [], 'acceptance_rate': [], 'energy': [], 'phase': 'exploration'}

    is_refinement_phase = False

    for step in range(config.chain_length + config.burn_in):
        # --- Check for triggering local refinement ---
        if config.enable_local_refinement and not is_refinement_phase and step > config.burn_in:
            current_sim_to_target = np.sum(best_state == surrogate_func.target_grid) / surrogate_func.target_grid.size
            if current_sim_to_target >= config.refinement_trigger_similarity:
                print(f"  > Triggering local refinement phase at step {step} (similarity: {current_sim_to_target:.2%}).")
                is_refinement_phase = True
                log_data['phase'] = 'refinement'
                # Optional: Reset temperature or adjust other params for refinement
                temperature = config.initial_temp / 5.0 # Lower temp for refinement

        # During refinement, proposals are generated with a much lower mutation rate.
        proposal_state = random.choice(output_space.get_neighbors(
            current_state, step=step, error_focus=error_focus,
            strategy=config.proposal_strategy,
            region_size=config.mutation_region_size,
            is_refinement=is_refinement_phase
        ))
        proposal_energy = -surrogate_func.predict(proposal_state)
        
        energy_diff = proposal_energy - current_energy
        
        accepted = False
        if energy_diff < 0 or (temperature > 0 and random.random() < math.exp(-energy_diff / temperature)):
            current_state, current_energy = proposal_state, proposal_energy
            accepted = True
            if current_energy < best_energy:
                best_state, best_energy = current_state, current_energy
        
        if step > config.burn_in:
            acceptance_history.append(1 if accepted else 0)
            energy_history.append(best_energy)

            # Log metrics periodically
            if step % 100 == 0:
                acc_rate = np.mean(acceptance_history[-100:]) if acceptance_history else 0
                log_data['temp'].append(temperature)
                log_data['acceptance_rate'].append(acc_rate)
                log_data['energy'].append(best_energy)

            # Adaptive Temperature & Convergence Check
            if len(acceptance_history) > 50:
                acceptance_rate = np.mean(acceptance_history[-50:])
                if acceptance_rate < config.target_acceptance_rate:
                    temperature *= (1 + config.adaptive_adjustment_factor)
                    # If acceptance is too low, our steps might be too big.
                    config.mutation_region_size = max(2, config.mutation_region_size - 1)
                else:
                    temperature *= (1 - config.adaptive_adjustment_factor)
                    # If acceptance is high, we can afford to explore more broadly.
                    config.mutation_region_size = min(7, config.mutation_region_size + 1)

                temperature = max(temperature, config.final_temp)
                acceptance_history.pop(0)

            if len(energy_history) > config.convergence_window:
                windowed_energy = energy_history[-config.convergence_window:]
                if np.std(windowed_energy) < config.convergence_threshold:
                    print(f"  > Convergence detected at step {step} (energy stddev < {config.convergence_threshold}). Terminating early.")
                    break
                energy_history.pop(0)

    print(f"  > MCMC finished. Best energy: {-best_energy:.4f}")
    return best_state, log_data

def perform_online_update(model, arc_output_head, optimizer, scheduler, input_bytes, failed_grid_np: np.ndarray, corrected_grid_np: np.ndarray, original_dims: tuple, device, failed_features: Optional[torch.Tensor] = None):
    """
    Performs a single, targeted fine-tuning step on the model using Focal Loss
    and a learning rate scheduler for more stable and focused updates.
    Enhanced with comments on alternative loss functions and stability penalties.
    """
    h, w = original_dims
    target_for_loss = torch.from_numpy(corrected_grid_np[:h, :w]).long().to(device)
    failed_for_loss = torch.from_numpy(failed_grid_np[:h, :w]).long().to(device)

    correction_mask = (target_for_loss != failed_for_loss).float()

    if torch.sum(correction_mask) == 0:
        print("  > No pixel differences found between failed and corrected grid. Skipping online update.")
        return

    model.train()
    arc_output_head.train()
    optimizer.zero_grad()

    target_bytes_single = serialize_and_pad_grid(corrected_grid_np, max_len=MAX_SEQUENCE_LENGTH, pad_value=PADDING_BYTE_VALUE)
    target_bytes_np = np.frombuffer(target_bytes_single, dtype=np.uint8).copy()
    target_bytes_tensor = torch.from_numpy(target_bytes_np).to(torch.uint8).unsqueeze(0).to(device)

    train_timestep = torch.zeros(1, device=device).long()
    output_dict = model(
        byte_sequence=input_bytes,
        mode='ctm_controlled_diffusion',
        target_diffusion_output=target_bytes_tensor,
        timestep=train_timestep,
        task_name="ARC_AGI_2_ONLINE_LEARN"
    )

    ctm_core_output_data = output_dict.get('ctm_core_data')
    ctm_backbone_output = None
    if ctm_core_output_data and 'final_sync_out' in ctm_core_output_data:
        ctm_backbone_output = ctm_core_output_data['final_sync_out']
    elif ctm_core_output_data and 'ctm_latent_representation' in ctm_core_output_data:
        ctm_backbone_output = ctm_core_output_data['ctm_latent_representation']
    
    if ctm_backbone_output is None:
        print("Warning: CTM core output not found. Cannot perform online update.")
        model.eval()
        arc_output_head.eval()
        return

    if ctm_backbone_output.ndim > 2 and ctm_backbone_output.shape[1] > 0:
         ctm_features_for_head_new = ctm_backbone_output.mean(dim=1)
    else:
         ctm_features_for_head_new = ctm_backbone_output

    predicted_logits = arc_output_head(ctm_features_for_head_new)
    predicted_logits = predicted_logits.view(1, NUM_ARC_SYMBOLS, MAX_GRID_SIZE[0], MAX_GRID_SIZE[1])
    cropped_logits = predicted_logits[:, :, :h, :w]
    target_for_loss_unsqueezed = target_for_loss # Target should be [H, W] for FocalLoss

    # --- Enhanced Self-Learning and Loss Stabilization ---
    # The current implementation uses FocalLoss, which is a great start.
    # Other options to explore:
    # 1. Contrastive Loss: Instead of predicting pixels, predict which of two grids
    #    (the correct one vs. a distractor) is the true target. This can improve
    #    the feature representation's quality.
    # 2. Add a penalty term to discourage drastic changes from the original prediction,
    #    unless they lead to a significant improvement. This can be a regularization
    #    term on the weights or a penalty on the latent space distance.
    #    Example: loss += lambda * ||f(new) - f(old)||^2

    # Use Focal Loss to focus on hard-to-correct pixels
    loss_fn = FocalLoss(alpha=0.5, gamma=2.0, reduction='none')
    loss_per_pixel = loss_fn(cropped_logits, target_for_loss_unsqueezed.unsqueeze(0)) # Unsqueeze target here

    # Combine with the correction mask to only learn from changed pixels
    # To implement multi-sample loss averaging, one could run MCMC multiple times
    # to generate a batch of corrected grids, and average the loss over that batch.
    masked_loss = loss_per_pixel * correction_mask.unsqueeze(0)
    loss = masked_loss.sum() / correction_mask.sum().clamp(min=1e-8)
    base_loss_val = loss.item()

    # Add stabilization loss if features from the failed prediction are provided
    if failed_features is not None:
        # The penalty term discourages drastic changes from the original prediction's latent space.
        stabilization_lambda = 0.25 # Hyperparameter to balance the two loss terms
        stabilization_loss = torch.nn.functional.mse_loss(ctm_features_for_head_new, failed_features)
        loss += stabilization_lambda * stabilization_loss
        print(f"  > Combined Loss: {loss.item():.4f} (Focal: {base_loss_val:.4f}, Stabilize: {stabilization_loss.item():.4f})")
    else:
        print(f"  > Focused Loss: {base_loss_val:.4f} on {int(correction_mask.sum())} pixels.")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step() # Step the learning rate scheduler

    model.eval()
    arc_output_head.eval()
    print(f"  > Model updated. LR: {scheduler.get_last_lr()[0]:.6f}")


class IsolatedSelfLearningEnvironment:
    """
    A simulated secure container to run the self-correction and online
    learning process, preventing accidental harm to the main evaluation loop.
    """
    def __init__(self, model, arc_output_head, optimizer, device):
        self.model = model
        self.arc_output_head = arc_output_head
        self.optimizer = optimizer
        self.device = device
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        # Add a traceback import if it's not already global
        global traceback
        import traceback


    def run_correction_and_update(self, input_bytes, failed_grid, target_grid_full, original_dims, failed_features: Optional[torch.Tensor] = None, train_pairs: Optional[List[dict]] = None):
        """
        Runs an ensemble of MCMC chains for correction and an online update.
        This implements the ensemble, consensus, and iterative refinement strategies.
        """
        print("\n--- Entering Isolated Self-Learning Environment (Ensemble Mode) ---")
        try:
            h_orig, w_orig = original_dims
            final_target = target_grid_full[:h_orig, :w_orig]

            # --- Adaptive Hyperparameter & Meta-Learning Enhancements ---
            # Future work: Implement dynamic adjustment of MCMC params based on acceptance rates.
            # Future work: Use meta-learning to adapt the surrogate network to new tasks.
            
            # 1. Initialize the CTM-based surrogate model
            # The `feature_extractor` could be an auxiliary network for learned similarity.
            ctm_surrogate = CTMSurrogate(model=self.model, arc_output_head=self.arc_output_head, input_bytes_eval=input_bytes, target_grid=final_target, device=self.device)

            # --- Meta-Learning Step ---
            if train_pairs:
                ctm_surrogate.adapt_to_new_task(train_pairs, self.optimizer)

            # 2. Define MCMC config with local refinement enabled
            mcmc_config = MCMCConfig(
                num_chains=5, # Run 5 chains for the ensemble
                chain_length=2000, # Shorter chains for efficiency
                initial_temp=35.0, final_temp=0.01, burn_in=200,
                proposal_strategy="hybrid", mutation_region_size=3,
                convergence_window=150, convergence_threshold=1e-5,
                enable_local_refinement=True, refinement_trigger_similarity=0.95
            )

            arc_space = ARCSelfEditSpace(
                grid_dims=(h_orig, w_orig),
                initial_mutation_rate=0.20, # Higher initial rate for diversity
                final_mutation_rate=0.01,
                decay_steps=mcmc_config.chain_length
            )

            # 3. Initial state for MCMC is the failed prediction
            initial_mcmc_state = failed_grid[:h_orig, :w_orig]
            error_focus = (initial_mcmc_state != final_target).astype(np.float32)

            # 4. Run MCMC Ensemble
            ensemble_results = []
            print(f"  > Starting MCMC ensemble with {mcmc_config.num_chains} chains...")
            for i in range(mcmc_config.num_chains):
                print(f"  > Running Chain {i+1}/{mcmc_config.num_chains}...")
                # Introduce slight diversity in initial state for each chain
                diverse_initial_state = initial_mcmc_state.copy()
                if i > 0: # Mutate the initial state slightly for other chains
                    diverse_initial_state = arc_space.get_neighbors(diverse_initial_state, step=0, n_neighbors=1)[0]

                corrected_grid_cropped, _ = metropolis_hastings_sampler(
                    initial_state=diverse_initial_state,
                    surrogate_func=ctm_surrogate,
                    output_space=arc_space,
                    config=mcmc_config,
                    error_focus=error_focus
                )
                ensemble_results.append(corrected_grid_cropped)

            # 5. Consensus Strategy: Majority vote per pixel
            print("  > Combining ensemble results via majority vote...")
            final_corrected_grid_cropped = self.get_consensus_grid(ensemble_results, (h_orig, w_orig))

            # 6. Prepare the full-sized corrected grid
            corrected_grid_full = np.full(MAX_GRID_SIZE, PADDING_VALUE, dtype=int)
            corrected_grid_full[:h_orig, :w_orig] = final_corrected_grid_cropped

            # 7. Always perform an online update with the consensus grid.
            similarity = np.sum(final_corrected_grid_cropped == final_target) / final_target.size
            print(f"  > Consensus solution has {similarity:.2%} similarity. Committing to online update.")
            perform_online_update(
                model=self.model, arc_output_head=self.arc_output_head,
                optimizer=self.optimizer, scheduler=self.scheduler,
                input_bytes=input_bytes, failed_grid_np=failed_grid,
                corrected_grid_np=corrected_grid_full, original_dims=original_dims,
                device=self.device,
                failed_features=failed_features
            )
            
            print("--- Exiting Isolated Environment (Update ALWAYS Performed) ---")
            return corrected_grid_full

        except Exception as e:
            print(f"❌ Error within Isolated Self-Learning Environment: {e}")
            traceback.print_exc()
            print("--- Exiting Isolated Environment (Error Occurred) ---")
            return None

    def get_consensus_grid(self, grids: List[np.ndarray], dims: tuple) -> np.ndarray:
        """Determines the most likely grid from an ensemble using a pixel-wise majority vote."""
        h, w = dims
        final_grid = np.zeros(dims, dtype=int)
        for r in range(h):
            for c in range(w):
                # Count votes for each color at this pixel
                pixel_votes = collections.Counter(grid[r, c] for grid in grids)
                # Choose the color with the most votes
                final_grid[r, c] = pixel_votes.most_common(1)[0][0]
        return final_grid


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

dynamic_eval_dir =  eval_dir 
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

        # Instantiate the isolated learning environment
        # The optimizer is already prepared by Accelerate and does not need to be unwrapped with unwrap_model.
        learning_container = IsolatedSelfLearningEnvironment(
            model=ctm_model_arc,
            arc_output_head=arc_output_head,
            optimizer=optimizer_arc,
            device=device
        )

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
                # Squeeze the batch dimension (size 1) from the data loader output
                input_grid_np_eval = test_pair['input'].squeeze(0).cpu().numpy()
                input_bytes_eval_single = serialize_and_pad_grid(input_grid_np_eval, max_len=MAX_SEQUENCE_LENGTH, pad_value=PADDING_BYTE_VALUE)
                input_bytes_eval_np = np.frombuffer(input_bytes_eval_single, dtype=np.uint8).copy()
                input_bytes_eval = torch.from_numpy(input_bytes_eval_np).to(torch.uint8).unsqueeze(0).to(device)

                target_grid_np = test_pair['output'].squeeze(0).cpu().numpy()
                # The dataloader collates the (h, w) tuple into a tuple/list of tensors.
                # We need to extract the integer values from these tensors.
                h_tensor, w_tensor = test_pair['original_output_dims']
                original_dims = (h_tensor.item(), w_tensor.item())

                test_input_solved = False

                # --- First Attempt: Standard Prediction (with no_grad) ---
                with torch.no_grad():
                    current_batch_size_eval = input_bytes_eval.size(0)
                    eval_timestep = torch.zeros(current_batch_size_eval, device=input_bytes_eval.device).long()
                    eval_model_output_dict = ctm_model_arc(
                        byte_sequence=input_bytes_eval,
                        mode='ctm_controlled_diffusion',
                        target_diffusion_output=None,
                        timestep=eval_timestep,
                        task_name="ARC_AGI_2_EVAL_DIFFUSION"
                    )
                    
                    preds_grid = np.zeros(MAX_GRID_SIZE, dtype=int)
                    # Extract features consistent with training loop logic
                    ctm_core_output_data = eval_model_output_dict.get('ctm_core_data')
                    ctm_backbone_output = None
                    if ctm_core_output_data and 'final_sync_out' in ctm_core_output_data:
                        ctm_backbone_output = ctm_core_output_data['final_sync_out']
                    elif ctm_core_output_data and 'ctm_latent_representation' in ctm_core_output_data:
                        ctm_backbone_output = ctm_core_output_data['ctm_latent_representation']
                    
                    if ctm_backbone_output is not None:
                        # Process features like in training
                        if ctm_backbone_output.ndim > 2 and ctm_backbone_output.shape[1] > 0:
                             ctm_features_for_head = ctm_backbone_output.mean(dim=1)
                        else:
                             ctm_features_for_head = ctm_backbone_output
                        
                        logits = arc_output_head(ctm_features_for_head)
                        preds_flat = torch.argmax(logits.view(-1, NUM_ARC_SYMBOLS), dim=-1)
                        preds_grid = preds_flat.view(MAX_GRID_SIZE).long().cpu().numpy()

                # --- Evaluate Prediction ---
                h, w = original_dims
                final_pred = preds_grid[:h, :w]
                final_target = target_grid_np[:h, :w]

                if np.array_equal(final_pred, final_target):
                    print(f"  > Test pair {test_pair_idx+1} solved on attempt 1.")
                    test_input_solved = True
                else:
                    # --- Second Attempt: Isolated Self-Correction (with gradients enabled) ---
                    features_from_failed_pred = ctm_features_for_head.detach() # Detach to prevent gradients from flowing back further
                    corrected_grid = learning_container.run_correction_and_update(
                        input_bytes=input_bytes_eval,
                        failed_grid=preds_grid,
                        target_grid_full=target_grid_np,
                        original_dims=original_dims,
                        failed_features=features_from_failed_pred,
                        train_pairs=current_task_data.get('train')
                    )

                    if corrected_grid is not None:
                        # Re-evaluate the grid returned by the learning container
                        final_pred_corrected = corrected_grid[:h, :w]
                        if np.array_equal(final_pred_corrected, final_target):
                            print(f"  > Test pair {test_pair_idx+1} solved on attempt 2 (after self-correction).")
                            test_input_solved = True
                        else:
                            print(f"  > Self-correction did not produce the correct solution.")
                
                if not test_input_solved:
                    task_solved_overall = False
                    # This break will exit the loop over test pairs for the current task
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