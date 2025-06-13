"""
Enhanced MCMC Layers with Blackbox Solver Integration and Correction Ratios

This module provides solutions for:
1. Opening the solver's blackbox for better control and inspection
2. Implementing correction ratios in acceptance rules for convergence guarantees
3. Supporting large neighborhood search algorithms with exact optimization oracles
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Tuple, Dict, Any, List
import numpy as np
import math
from dataclasses import dataclass

from .fenchel_young_mcmc import (
    MCMCConfig, DiscreteOutputSpace, BinaryHypercube, TopKPolytope,
    TemperatureScheduler
)
from .modules import SuperLinear
from .utils import compute_normalized_entropy
from .mcmc_interpretability_solver import (BlackBoxSolver, MCMCInterpretabilityHook, ReasoningChain, ThoughtStep)


class ExactOptimizationOracle:
    """Exact optimization oracle for large neighborhood search with BlackBoxSolver integration"""
    
    def __init__(self, output_space: DiscreteOutputSpace, phi_network: Optional[nn.Module] = None, model: Optional[nn.Module] = None):
        self.output_space = output_space
        
        # Initialize BlackBoxSolver if model is provided
        if model is not None:
            self.blackbox_solver = BlackBoxSolver(model)
        else:
            self.blackbox_solver = None
        self.phi_network = phi_network
        self.solver_state = {
            'last_solution': None,
            'last_objective_value': None,
            'num_evaluations': 0,
            'optimization_history': []
        }
    
    def solve(self, theta: torch.Tensor, neighborhood: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Solve exactly within the given neighborhood or entire space"""
        search_space = neighborhood if neighborhood is not None else self.output_space.output_space
        
        best_solution = None
        best_value = float('-inf')
        
        for candidate in search_space:
            # Compute objective: θ^T y + φ(y)
            objective_value = torch.dot(theta.squeeze(0), candidate)
            if self.phi_network is not None:
                objective_value += self.phi_network(candidate).squeeze()
            
            if objective_value > best_value:
                best_value = objective_value
                best_solution = candidate
            
            self.solver_state['num_evaluations'] += 1
        
        self.solver_state['last_solution'] = best_solution
        self.solver_state['last_objective_value'] = float(best_value)
        self.solver_state['optimization_history'].append({
            'solution': best_solution.clone(),
            'value': float(best_value),
            'search_space_size': len(search_space)
        })
        
        return best_solution
    
    def get_solver_state(self) -> Dict[str, Any]:
        return self.solver_state.copy()
    
    def set_solver_parameters(self, params: Dict[str, Any]) -> None:
        # For exact oracle, we might set convergence tolerances, etc.
        if 'reset_history' in params and params['reset_history']:
            self.solver_state['optimization_history'] = []
            self.solver_state['num_evaluations'] = 0


class CorrectionRatioMCMC(nn.Module):
    """
    Enhanced MCMC with correction ratios for guaranteed convergence
    to the correct stationary distribution
    """
    
    def __init__(self, 
                 output_space: DiscreteOutputSpace,
                 config: MCMCConfig,
                 phi_network: Optional[nn.Module] = None,
                 exact_oracle: Optional[ExactOptimizationOracle] = None):
        super().__init__()
        
        self.output_space = output_space
        self.config = config
        self.phi_network = phi_network
        self.exact_oracle = exact_oracle
        
        # Temperature scheduler
        self.temp_scheduler = self._create_temperature_scheduler()
        
        # Persistent states and correction tracking
        self.persistent_states = None
        self.correction_ratios = []
        self.step_count = 0
        
        # Convergence diagnostics
        self.acceptance_rates = []
        self.effective_sample_sizes = []
        self.autocorrelation_times = []
        
        # Blackbox solver integration
        self.solver_diagnostics = []
    
    def _create_temperature_scheduler(self) -> Callable[[int], float]:
        """Create temperature scheduler based on config"""
        if self.config.temperature_schedule == "geometric":
            return TemperatureScheduler.geometric(
                self.config.initial_temp, 
                self.config.decay_rate, 
                self.config.final_temp
            )
        elif self.config.temperature_schedule == "linear":
            return TemperatureScheduler.linear(
                self.config.initial_temp,
                self.config.final_temp,
                self.config.chain_length
            )
        else:
            return TemperatureScheduler.constant(self.config.final_temp)
    
    def phi_function(self, state: torch.Tensor) -> torch.Tensor:
        """Structure function φ(y)"""
        if self.phi_network is not None:
            return self.phi_network(state)
        return torch.tensor(0.0)
    
    def _compute_boundary_correction(self, current: torch.Tensor, proposal: torch.Tensor) -> float:
        """Compute correction for boundary effects in discrete spaces"""
        # For binary hypercube, boundary effects occur at corners
        if isinstance(self.output_space, BinaryHypercube):
            current_boundary_score = torch.sum(torch.abs(current - 0.5))
            proposal_boundary_score = torch.sum(torch.abs(proposal - 0.5))
            
            # States closer to boundaries have fewer neighbors
            # Correction factor accounts for this asymmetry
            return float(torch.exp(0.1 * (current_boundary_score - proposal_boundary_score)))
        
        return 1.0  # No correction for other spaces by default
    
    def enhanced_acceptance_ratio(self,
                                current: torch.Tensor,
                                proposal: torch.Tensor,
                                theta: torch.Tensor,
                                temperature: float,
                                strategy_name: str, # New parameter
                                strategy_params: Dict[str, Any] # New parameter
                                ) -> float:
        """
        Enhanced Metropolis-Hastings acceptance term p(k) = α_s * exp(Δ(k)/t_k).
        This value is then compared with U ~ Uniform(0,1) in the sampling loop.
        α_s is the full correction factor from compute_correction_ratio.
        """
        # Energy difference: E(proposal) - E(current)
        # Original paper: Δ(k) = <θ, y'> + φ(y') - <θ, y(k)> - φ(y(k))
        current_energy = torch.dot(theta.squeeze(0), current) + self.phi_function(current).squeeze()
        proposal_energy = torch.dot(theta.squeeze(0), proposal) + self.phi_function(proposal).squeeze()
        energy_diff = proposal_energy - current_energy
        
        # Correction factor α_s = [ |Q(y)|/|Q(y')| ] * [ q_s(y',y) / q_s(y,y') ]
        # Theta is passed for signature consistency, though not directly used by the new compute_correction_ratio
        correction_factor = self.compute_correction_ratio(current, proposal, theta, strategy_name, strategy_params)
        
        if correction_factor < 0: # Should ideally not happen with valid proposal probabilities
            # print(f"Warning: Negative correction_factor encountered: {correction_factor}")
            correction_factor = 0.0 # Treat as 0 probability if something went wrong

        acceptance_term_pk: float
        if temperature <= 1e-9: # Avoid division by zero or extreme values with very low/zero temp
            if energy_diff <= 0: # If energy decreases or stays same
                # If correction_factor is very large (e.g. inf), this will be inf.
                # If correction_factor is 0, this will be 0.
                acceptance_term_pk = float('inf') if correction_factor > 1e-9 else 0.0 # Effectively accept if possible and correction_factor > 0
            else: # If energy increases, reject at zero temperature
                acceptance_term_pk = 0.0
        else:
            # acceptance_term_pk = float(correction_factor * torch.exp(energy_diff / temperature))
            # Ensure energy_diff is a scalar tensor if it's not already
            if not isinstance(energy_diff, torch.Tensor):
                energy_diff_tensor = torch.tensor(energy_diff, device=theta.device)
            else:
                energy_diff_tensor = energy_diff

            exp_term = torch.exp(energy_diff_tensor / temperature)
            acceptance_term_pk = float(correction_factor * exp_term)

        # Ensure the returned value is non-negative.
        # The comparison `random.random() < acceptance_term_pk` handles the min(1, pk) implicitly.
        return max(0.0, acceptance_term_pk)
    
    def large_neighborhood_search_step(self, 
                                     current_state: torch.Tensor, 
                                     theta: torch.Tensor, 
                                     neighborhood_size: int = 5) -> torch.Tensor:
        """
        Large neighborhood search using exact optimization oracle
        """
        if self.exact_oracle is None:
            # Fallback to regular neighborhood if no oracle available
            neighbors = self.output_space.get_neighbors(current_state, self.config.neighborhood_radius)
            return neighbors[np.random.randint(len(neighbors))] if neighbors else current_state
        
        # Generate large neighborhood
        large_neighborhood = []
        
        # Start with immediate neighbors
        immediate_neighbors = self.output_space.get_neighbors(current_state, 1)
        large_neighborhood.extend(immediate_neighbors)
        
        # Add random states to expand neighborhood
        while len(large_neighborhood) < neighborhood_size:
            random_state = self.output_space.random_state()
            if not any(torch.allclose(random_state, existing) for existing in large_neighborhood):
                large_neighborhood.append(random_state)
        
        # Use exact oracle to find best solution in neighborhood
        best_solution = self.exact_oracle.solve(theta, large_neighborhood)
        
        # Store solver diagnostics
        solver_state = self.exact_oracle.get_solver_state()
        self.solver_diagnostics.append(solver_state)
        
        return best_solution
    
    def sample_chain_corrected(self,
                               theta: torch.Tensor,
                               chain_id: int = 0,
                               target_state: Optional[torch.Tensor] = None,
                               use_large_neighborhood_step: bool = False) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """
        Run single MCMC chain using enhanced acceptance ratio, optional LNS,
        and mixed neighborhood proposal strategies (Algorithm 2 from paper).
        """
        if self.config.initialization_method == "persistent" and self.persistent_states is not None and \
           chain_id < len(self.persistent_states) and self.persistent_states[chain_id] is not None:
            current_state = self.persistent_states[chain_id].clone()
        elif self.config.initialization_method == "data_based" and target_state is not None:
            current_state = target_state.clone()
        else:
            current_state = self.output_space.random_state()

        samples = []
        acceptances = 0
        total_steps_for_chain = self.config.chain_length + self.config.burn_in
        
        current_chain_acceptance_terms = [] # Store p_k values for diagnostics

        for step_idx in range(total_steps_for_chain):
            temperature = self.temp_scheduler(step_idx)
            
            proposal = None
            chosen_strategy_name = "unknown"
            strategy_params = {}

            # Determine if LNS step should be taken (specific to LargeNeighborhoodSearchMCMC subclass)
            # The 'use_large_neighborhood_step' flag is passed down.
            # The actual LNS attributes like lns_frequency are on the LNSMCMC instance.
            is_lns_step_type = isinstance(self, LargeNeighborhoodSearchMCMC)
            
            perform_lns_this_iteration = False
            if use_large_neighborhood_step and is_lns_step_type and self.exact_oracle:
                lns_freq = getattr(self, 'lns_frequency', 10) # Default if not present
                if lns_freq > 0 and (step_idx + 1) % lns_freq == 0:
                    perform_lns_this_iteration = True
            
            if perform_lns_this_iteration:
                lns_hood_size = getattr(self, 'lns_neighborhood_size', 5) # Default
                proposal = self.large_neighborhood_search_step(current_state, theta, lns_hood_size)
                chosen_strategy_name = "LNS"
                # For LNS, strategy_params might be empty or indicate LNS-specifics if needed by correction_ratio
                strategy_params = {'lns_generated': True} 
            else:
                # Algorithm 2: Mixing neighborhood systems for standard proposals
                available_strategies = self.output_space.get_available_neighborhood_strategies(current_state)
                if not available_strategies:
                    # If no strategies, chain is stuck with local moves. Stay put.
                    if step_idx >= self.config.burn_in:
                        samples.append(current_state.clone())
                    current_chain_acceptance_terms.append(0.0) # No move attempted
                    continue

                chosen_strategy_name = random.choice(available_strategies)
                
                # Derive strategy_params from chosen_strategy_name
                if chosen_strategy_name.startswith("radius_"):
                    try:
                        strategy_params['radius'] = int(chosen_strategy_name.split("_")[1])
                    except (IndexError, ValueError):
                        strategy_params['radius'] = self.config.neighborhood_radius # Fallback
                elif chosen_strategy_name.startswith("swaps_"):
                    try:
                        strategy_params['num_swaps'] = int(chosen_strategy_name.split("_")[1])
                    except (IndexError, ValueError):
                        strategy_params['num_swaps'] = 1 # Fallback for swaps
                else: # Default or other strategies might use default radius
                    strategy_params['radius'] = self.config.neighborhood_radius
                
                neighbors = self.output_space.get_neighbors(current_state, chosen_strategy_name, **strategy_params)
                if not neighbors:
                    if step_idx >= self.config.burn_in: # Still add current state if no proposal
                        samples.append(current_state.clone())
                    current_chain_acceptance_terms.append(0.0) # No move proposed
                    continue
                proposal = random.choice(neighbors)
            
            if proposal is None: # Should not happen if logic above is correct
                if step_idx >= self.config.burn_in:
                    samples.append(current_state.clone())
                current_chain_acceptance_terms.append(0.0)
                continue

            # Calculate acceptance term p_k = α_s * exp(Δ_k / t_k)
            # The enhanced_acceptance_ratio now returns this p_k directly.
            acceptance_term_pk = self.enhanced_acceptance_ratio(
                current_state, proposal, theta, temperature,
                chosen_strategy_name, strategy_params
            )
            current_chain_acceptance_terms.append(acceptance_term_pk)
            
            if random.random() < min(1.0, acceptance_term_pk): # Explicit min(1,.) for clarity with M-H
                current_state = proposal
                acceptances += 1
            
            if step_idx >= self.config.burn_in:
                samples.append(current_state.clone())
        
        if self.persistent_states is None or len(self.persistent_states) != self.config.num_chains:
             self.persistent_states = [self.output_space.random_state() for _ in range(self.config.num_chains)]
        self.persistent_states[chain_id] = current_state.clone()
        
        avg_pk_for_chain = np.mean(current_chain_acceptance_terms) if current_chain_acceptance_terms else 0.0
        
        stats = {
            'acceptance_rate': acceptances / total_steps_for_chain if total_steps_for_chain > 0 else 0.0,
            'final_temperature': temperature, # Will be the last temperature used
            'chain_length_collected': len(samples),
            'avg_acceptance_term_pk': float(avg_pk_for_chain) # Average of p_k values
        }
        return samples, stats

    def estimate_expectation_with_corrections(self,
                                            theta: torch.Tensor,
                                            target_state: Optional[torch.Tensor] = None,
                                            use_large_neighborhood: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Estimate E[Y] under Gibbs distribution using multiple MCMC chains with corrections.
        Handles batched theta and target_state.
        """
        original_theta_ndim = theta.ndim
        if original_theta_ndim == 1:
            theta_batch = theta.unsqueeze(0)
            if target_state is not None:
                # Assuming target_state is (dim) if theta is (dim)
                target_state_batch = target_state.unsqueeze(0) if target_state.ndim == 1 else target_state
            else:
                target_state_batch = None
        elif original_theta_ndim == 2:
            theta_batch = theta
            if target_state is not None:
                if target_state.ndim == 2 and target_state.shape[0] == theta.shape[0]: # Batched target
                    target_state_batch = target_state
                elif target_state.ndim == 1: # Unbatched target, will be broadcasted effectively
                    target_state_batch = target_state
                else:
                    raise ValueError(f"target_state shape {target_state.shape} incompatible with batched theta {theta.shape}")
            else:
                target_state_batch = None
        else:
            raise ValueError(f"theta must be 1D or 2D, got {original_theta_ndim}D")

        batch_size = theta_batch.shape[0]
        batch_expectations = []
        batch_overall_stats_list = []

        # Ensure persistent_states is initialized correctly and on the correct device
        current_device = theta_batch.device
        if self.persistent_states is None or len(self.persistent_states) != self.config.num_chains:
            self.persistent_states = [self.output_space.random_state().to(current_device) for _ in range(self.config.num_chains)]
        else: # Ensure existing states are on the correct device
            for i in range(len(self.persistent_states)):
                if self.persistent_states[i].device != current_device:
                    self.persistent_states[i] = self.persistent_states[i].to(current_device)


        for b_idx in range(batch_size):
            current_theta_slice = theta_batch[b_idx]
            current_target_slice = None
            if target_state_batch is not None:
                if target_state_batch.ndim == 2: # Batched target
                    current_target_slice = target_state_batch[b_idx]
                else: # Unbatched target (ndim == 1) or None
                    current_target_slice = target_state_batch

            all_samples_for_item = []
            all_chain_stats_for_item = []

            for chain_id in range(self.config.num_chains):
                is_lns_sampler = isinstance(self, LargeNeighborhoodSearchMCMC)
                samples, chain_stats_dict = self.sample_chain_corrected(
                    current_theta_slice,
                    chain_id,
                    current_target_slice,
                    use_large_neighborhood_step=(use_large_neighborhood and is_lns_sampler)
                )
                all_samples_for_item.extend(samples)
                all_chain_stats_for_item.append(chain_stats_dict)

            if not all_samples_for_item:
                dim_fallback = self.output_space.dimension if hasattr(self.output_space, 'dimension') else current_theta_slice.shape[0]
                zero_fallback_item = torch.zeros(dim_fallback, device=current_device, dtype=current_theta_slice.dtype)
                batch_expectations.append(zero_fallback_item)
                batch_overall_stats_list.append({
                    'error': 'No samples collected for this batch item',
                    'num_samples': 0,
                    'avg_acceptance_rate': 0.0,
                    'avg_acceptance_term_pk': 0.0,
                    'sample_entropy': 0.0,
                    'chain_stats': []
                })
                # print(f"Warning: No MCMC samples collected for batch item {b_idx}. Returning zeros for this item.")
                continue

            expectation_item = torch.mean(torch.stack(all_samples_for_item), dim=0)
            batch_expectations.append(expectation_item)

            avg_acceptance_item = np.mean([s.get('acceptance_rate', 0.0) for s in all_chain_stats_for_item]) if all_chain_stats_for_item else 0.0
            avg_pk_item = np.mean([s.get('avg_acceptance_term_pk', 0.0) for s in all_chain_stats_for_item]) if all_chain_stats_for_item else 0.0
            
            sample_stack_for_entropy_item = torch.stack(all_samples_for_item).detach().cpu()
            sample_entropy_item = compute_normalized_entropy(sample_stack_for_entropy_item)

            combined_stats_item = {
                'num_samples': len(all_samples_for_item),
                'avg_acceptance_rate': float(avg_acceptance_item),
                'avg_acceptance_term_pk': float(avg_pk_item),
                'sample_entropy': sample_entropy_item.tolist(), # Store list of per-sample entropies
                'chain_stats': all_chain_stats_for_item
            }
            batch_overall_stats_list.append(combined_stats_item)

        final_expectation = torch.stack(batch_expectations)

        total_num_samples_agg = sum(s['num_samples'] for s in batch_overall_stats_list)
        
        valid_acceptance_rates = [s['avg_acceptance_rate'] for s in batch_overall_stats_list if s.get('num_samples',0) > 0]
        avg_acceptance_rate_agg = np.mean(valid_acceptance_rates) if valid_acceptance_rates else 0.0
        
        valid_pk_terms = [s['avg_acceptance_term_pk'] for s in batch_overall_stats_list if s.get('num_samples',0) > 0]
        avg_acceptance_term_pk_agg = np.mean(valid_pk_terms) if valid_pk_terms else 0.0

        valid_entropies = [s['sample_entropy'] for s in batch_overall_stats_list if s.get('num_samples',0) > 0]
        all_individual_entropies = [e for sublist in valid_entropies for e in sublist]
        avg_sample_entropy_agg = np.mean(all_individual_entropies) if all_individual_entropies else 0.0

        final_combined_stats = {
            'num_total_samples': total_num_samples_agg,
            'avg_acceptance_rate_across_batch': float(avg_acceptance_rate_agg),
            'avg_acceptance_term_pk_across_batch': float(avg_acceptance_term_pk_agg),
            'avg_sample_entropy_across_batch': float(avg_sample_entropy_agg),
            'per_batch_item_stats': batch_overall_stats_list
        }

        if original_theta_ndim == 1:
            final_expectation = final_expectation.squeeze(0)
            # If squeezing expectation, should we also simplify stats if batch_size was 1?
            # For now, stats structure remains batched even if input was 1D.

        return final_expectation, final_combined_stats
    
    def forward(self, 
                theta: torch.Tensor, 
                target: torch.Tensor,
                use_large_neighborhood: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with enhanced MCMC and correction ratios
        """
        # Estimate expectation with corrections
        expectation, stats = self.estimate_expectation_with_corrections(
            theta, use_large_neighborhood
        )
        
        # Fenchel-Young gradient with correction
        gradient = expectation - target
        
        # Enhanced loss computation
        loss = torch.dot(theta, expectation) - torch.dot(theta, target)
        
        # Store correction ratios for monitoring
        if 'avg_correction_rate' in stats:
            self.correction_ratios.append(stats['avg_correction_rate'])
        
        self.step_count += 1
        
        return loss, {**stats, 'gradient': gradient}


class LargeNeighborhoodSearchMCMC(CorrectionRatioMCMC):
    """
    MCMC with integrated large neighborhood search algorithms
    """
    
    def __init__(self, 
                 output_space: DiscreteOutputSpace,
                 config: MCMCConfig,
                 phi_network: Optional[nn.Module] = None,
                 lns_frequency: int = 10,
                 lns_neighborhood_size: int = 20):
        
        # Create exact optimization oracle
        exact_oracle = ExactOptimizationOracle(output_space, phi_network)
        
        super().__init__(output_space, config, phi_network, exact_oracle)
        
        self.lns_frequency = lns_frequency
        self.lns_neighborhood_size = lns_neighborhood_size
        
        # LNS-specific tracking
        self.lns_improvements = []
        self.lns_statistics = []


# Enhanced CTM Integration
class EnhancedCTMFenchelYoungIntegration(nn.Module):
    """
    Enhanced CTM integration with corrected MCMC and large neighborhood search
    """
    
    def __init__(self,
                 input_dim: int,
                 output_space: DiscreteOutputSpace,
                 mcmc_config: MCMCConfig,
                 hidden_dim: int = 256,
                 num_thought_steps: int = 5,
                 use_large_neighborhood_search: bool = True,
                 lns_frequency: int = 10):
        super().__init__()
        
        # CTM thinking network
        self.thought_network = nn.Sequential(
            SuperLinear(input_dim, hidden_dim),
            nn.ReLU(),
            SuperLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            SuperLinear(hidden_dim, output_space.dimension)
        )
        
        # Enhanced structure function
        self.phi_network = nn.Sequential(
            nn.Linear(output_space.dimension, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Choose MCMC implementation
        if use_large_neighborhood_search:
            self.mcmc_sampler = LargeNeighborhoodSearchMCMC(
                output_space=output_space,
                config=mcmc_config,
                phi_network=self.phi_network,
                lns_frequency=lns_frequency
            )
        else:
            self.mcmc_sampler = CorrectionRatioMCMC(
                output_space=output_space,
                config=mcmc_config,
                phi_network=self.phi_network
            )
        
        self.num_thought_steps = num_thought_steps
        self.output_space = output_space
        self.use_large_neighborhood_search = use_large_neighborhood_search
    
    def think(self, x: torch.Tensor, num_steps: Optional[int] = None) -> List[torch.Tensor]:
        """CTM thinking process"""
        if num_steps is None:
            num_steps = self.num_thought_steps
        
        thoughts = []
        current_thought = x
        
        for _ in range(num_steps):
            current_thought = self.thought_network(current_thought)
            thoughts.append(current_thought)
        
        return thoughts
    
    def forward(self, 
                x: torch.Tensor, 
                target: torch.Tensor,
                return_thoughts: bool = False,
                return_diagnostics: bool = False) -> Dict[str, Any]:
        """
        Enhanced forward pass with corrected MCMC
        """
        # Generate thoughts
        thoughts = self.think(x)
        final_theta = thoughts[-1]
        
        # Compute loss with enhanced MCMC
        loss, mcmc_stats = self.mcmc_sampler(final_theta, target)
        
        result = {
            'loss': loss,
            'mcmc_stats': mcmc_stats,
            'final_theta': final_theta
        }
        
        if return_thoughts:
            result['thoughts'] = thoughts
        
        if return_diagnostics:
            result['solver_diagnostics'] = self.mcmc_sampler.solver_diagnostics
            result['correction_ratios'] = self.mcmc_sampler.correction_ratios
        
        return result

