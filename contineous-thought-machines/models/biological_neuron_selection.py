"""
Biologically-Inspired Neuron Selection for Continuous Thought Machines

This module implements various biologically-inspired neuron selection strategies
that can replace the current random selection methods with more principled approaches
based on computational neuroscience and biological neural network principles.

Key biological principles implemented:
1. Hebbian Learning: "Neurons that fire together, wire together"
2. Synaptic Plasticity: Adaptive strength based on usage patterns
3. Competitive Learning: Winner-take-all and lateral inhibition
4. Homeostatic Regulation: Maintaining optimal activity levels
5. Evolutionary Selection: Fitness-based neuron survival
6. Spike-Timing Dependent Plasticity (STDP): Temporal correlation learning
7. Neural Criticality: Self-organized criticality in neural networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import math


@dataclass
class BiologicalSelectionConfig:
    """Configuration for biological neuron selection methods."""
    selection_type: str = 'hebbian'
    sparsity_ratio: float = 0.5
    learning_rate: float = 0.01
    decay_rate: float = 0.99
    competition_strength: float = 1.0
    homeostatic_target: float = 0.1
    plasticity_window: int = 100
    criticality_target: float = 1.0
    use_temporal_dynamics: bool = True
    adaptation_rate: float = 0.001


class BiologicalNeuronSelector:
    """
    Biologically-inspired neuron selection for CTM models.
    
    This class implements multiple selection strategies based on biological
    neural network principles, providing more principled alternatives to
    random neuron selection.
    """
    
    def __init__(self, config: BiologicalSelectionConfig):
        self.config = config
        self.selection_type = config.selection_type
        
        # Hebbian learning state
        self.correlation_matrix = None
        self.activation_history = deque(maxlen=config.plasticity_window)
        
        # Plasticity tracking
        self.synaptic_strengths = {}
        self.plasticity_traces = {}
        self.weight_change_history = {}
        
        # Competitive learning state
        self.inhibition_strengths = {}
        self.winner_history = deque(maxlen=config.plasticity_window)
        
        # Homeostatic regulation
        self.activity_targets = {}
        self.intrinsic_excitability = {}
        self.homeostatic_scaling = {}
        
        # Evolutionary selection
        self.fitness_scores = {}
        self.generation_count = 0
        self.selection_pressure = 1.0
        
        # STDP (Spike-Timing Dependent Plasticity)
        self.spike_times = {}
        self.stdp_window = 20  # ms
        
        # Neural criticality
        self.avalanche_sizes = deque(maxlen=1000)
        self.branching_parameter = 1.0
        
        # Performance tracking
        self.selection_history = []
        self.performance_metrics = defaultdict(list)
        
    def select_neurons(self, 
                      activations: torch.Tensor,
                      targets: Optional[torch.Tensor] = None,
                      weights: Optional[torch.Tensor] = None,
                      weight_changes: Optional[torch.Tensor] = None,
                      learning_rates: Optional[torch.Tensor] = None,
                      top_k: Optional[int] = None,
                      layer_name: str = "default") -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Main selection method that routes to specific biological strategies.
        
        Args:
            activations: Neuron activation patterns [batch_size, num_neurons]
            targets: Target patterns for supervised selection
            weights: Current neuron weights
            weight_changes: Recent weight changes for plasticity-based selection
            learning_rates: Learning rates per neuron
            top_k: Number of neurons to select
            layer_name: Name of the layer for tracking
            
        Returns:
            Tuple of (selected_indices, selection_metadata)
        """
        if top_k is None:
            top_k = int(activations.shape[-1] * (1.0 - self.config.sparsity_ratio))
            
        # Route to appropriate selection method
        if self.selection_type == 'hebbian':
            return self._hebbian_selection(activations, targets, top_k, layer_name)
        elif self.selection_type == 'plasticity':
            return self._plasticity_selection(activations, weight_changes, learning_rates, top_k, layer_name)
        elif self.selection_type == 'competitive':
            return self._competitive_selection(activations, top_k, layer_name)
        elif self.selection_type == 'homeostatic':
            return self._homeostatic_selection(activations, top_k, layer_name)
        elif self.selection_type == 'evolutionary':
            return self._evolutionary_selection(activations, targets, top_k, layer_name)
        elif self.selection_type == 'stdp':
            return self._stdp_selection(activations, top_k, layer_name)
        elif self.selection_type == 'criticality':
            return self._criticality_selection(activations, top_k, layer_name)
        elif self.selection_type == 'multi_objective':
            return self._multi_objective_selection(activations, targets, weights, top_k, layer_name)
        else:
            raise ValueError(f"Unknown selection type: {self.selection_type}")
    
    def _hebbian_selection(self, activations: torch.Tensor, targets: Optional[torch.Tensor], 
                          top_k: int, layer_name: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Hebbian learning: "Neurons that fire together, wire together"
        
        Selects neurons based on correlation patterns with targets or co-activation.
        """
        batch_size, num_neurons = activations.shape
        device = activations.device
        
        if targets is not None:
            # Supervised Hebbian: correlation with targets
            if targets.dim() == 1:
                targets = targets.unsqueeze(0).expand(batch_size, -1)
            
            # Compute correlation between each neuron and target
            activations_centered = activations - activations.mean(dim=0, keepdim=True)
            targets_centered = targets - targets.mean(dim=0, keepdim=True)
            
            # Correlation coefficient for each neuron with target
            correlations = []
            for i in range(num_neurons):
                neuron_acts = activations_centered[:, i]
                if targets.shape[1] == 1:
                    target_vals = targets_centered[:, 0]
                else:
                    target_vals = targets_centered[:, i % targets.shape[1]]
                
                corr = torch.corrcoef(torch.stack([neuron_acts, target_vals]))[0, 1]
                correlations.append(corr if not torch.isnan(corr) else torch.tensor(0.0, device=device))
            
            hebbian_scores = torch.stack(correlations)
        else:
            # Unsupervised Hebbian: co-activation patterns
            # Update correlation matrix with exponential moving average
            if batch_size > 1:
                current_corr = torch.corrcoef(activations.T)
                current_corr = torch.nan_to_num(current_corr, 0.0)
                
                if self.correlation_matrix is None:
                    self.correlation_matrix = current_corr
                else:
                    self.correlation_matrix = (self.config.decay_rate * self.correlation_matrix +
                                             (1 - self.config.decay_rate) * current_corr)

            if self.correlation_matrix is not None:
                # Select neurons with high average correlation (excluding self-correlation)
                mask = ~torch.eye(num_neurons, dtype=torch.bool, device=device)
                avg_correlations = torch.sum(torch.abs(self.correlation_matrix) * mask, dim=1) / (num_neurons - 1)
                hebbian_scores = avg_correlations
            else:
                # Fallback for batch_size <= 1 and no existing correlation matrix:
                # Use activation magnitude as a proxy for importance.
                hebbian_scores = torch.mean(torch.abs(activations), dim=0)
        
        # Store activation history for future use
        self.activation_history.append(activations.detach().cpu())
        
        # Select top-k neurons
        selected_indices = torch.topk(torch.abs(hebbian_scores), top_k).indices
        
        metadata = {
            'selection_type': 'hebbian',
            'hebbian_scores': hebbian_scores,
            'correlation_matrix': self.correlation_matrix,
            'mean_correlation': torch.mean(torch.abs(hebbian_scores)).item()
        }
        
        return selected_indices, metadata
    
    def _plasticity_selection(self, activations: torch.Tensor, weight_changes: Optional[torch.Tensor],
                             learning_rates: Optional[torch.Tensor], top_k: int, 
                             layer_name: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Synaptic plasticity-based selection.
        
        Selects neurons showing the most adaptive changes and learning efficiency.
        """
        num_neurons = activations.shape[-1]
        device = activations.device
        
        if weight_changes is None:
            # Estimate plasticity from activation patterns
            if len(self.activation_history) > 1:
                prev_activations = self.activation_history[-1].to(device)
                activation_changes = torch.norm(activations - prev_activations, dim=0)
            else:
                activation_changes = torch.norm(activations, dim=0)
            plasticity_scores = activation_changes
        else:
            # Use actual weight changes
            if weight_changes.dim() > 1:
                weight_change_magnitude = torch.norm(weight_changes, dim=0)
            else:
                weight_change_magnitude = torch.abs(weight_changes)
            
            # Combine with learning rates if available
            if learning_rates is not None:
                plasticity_scores = weight_change_magnitude * learning_rates
            else:
                plasticity_scores = weight_change_magnitude
        
        # Update plasticity traces with exponential decay
        layer_key = f"{layer_name}_plasticity"
        if layer_key not in self.plasticity_traces:
            self.plasticity_traces[layer_key] = torch.zeros(num_neurons, device=device)
        
        self.plasticity_traces[layer_key] = (self.config.decay_rate * self.plasticity_traces[layer_key] + 
                                           (1 - self.config.decay_rate) * plasticity_scores)
        
        # Select neurons with highest plasticity
        selected_indices = torch.topk(self.plasticity_traces[layer_key], top_k).indices
        
        metadata = {
            'selection_type': 'plasticity',
            'plasticity_scores': plasticity_scores,
            'plasticity_traces': self.plasticity_traces[layer_key],
            'mean_plasticity': torch.mean(plasticity_scores).item()
        }
        
        return selected_indices, metadata
    
    def _competitive_selection(self, activations: torch.Tensor, top_k: int, 
                              layer_name: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Competitive learning with lateral inhibition.
        
        Implements winner-take-all dynamics where strong neurons inhibit weaker ones.
        """
        batch_size, num_neurons = activations.shape
        device = activations.device
        
        # Apply lateral inhibition
        max_activation = torch.max(activations, dim=-1, keepdim=True)[0]
        mean_activation = torch.mean(activations, dim=-1, keepdim=True)
        
        # Inhibition strength based on distance from maximum
        inhibition = self.config.competition_strength * (max_activation - activations)
        inhibited_activations = activations - inhibition
        
        # Soft winner-take-all: use softmax with temperature
        temperature = 1.0 / (1.0 + self.config.competition_strength)
        competition_scores = F.softmax(inhibited_activations / temperature, dim=-1)
        
        # Average across batch
        avg_competition_scores = torch.mean(competition_scores, dim=0)
        
        # Update winner history
        winners = torch.argmax(competition_scores, dim=-1)
        self.winner_history.extend(winners.cpu().tolist())
        
        # Select top-k based on competition scores
        selected_indices = torch.topk(avg_competition_scores, top_k).indices
        
        metadata = {
            'selection_type': 'competitive',
            'competition_scores': avg_competition_scores,
            'inhibited_activations': inhibited_activations,
            'winner_frequency': self._compute_winner_frequency(num_neurons)
        }
        
        return selected_indices, metadata
    
    def _homeostatic_selection(self, activations: torch.Tensor, top_k: int, 
                              layer_name: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Homeostatic regulation selection.
        
        Maintains optimal activity levels by adjusting neuron excitability.
        """
        batch_size, num_neurons = activations.shape
        device = activations.device
        
        # Initialize homeostatic parameters if needed
        layer_key = f"{layer_name}_homeostatic"
        if layer_key not in self.activity_targets:
            self.activity_targets[layer_key] = torch.full((num_neurons,), 
                                                        self.config.homeostatic_target, 
                                                        device=device)
            self.intrinsic_excitability[layer_key] = torch.ones(num_neurons, device=device)
        
        # Compute current activity levels
        current_activity = torch.mean(torch.abs(activations), dim=0)
        
        # Update intrinsic excitability based on activity deviation
        activity_error = self.activity_targets[layer_key] - current_activity
        self.intrinsic_excitability[layer_key] += self.config.adaptation_rate * activity_error
        self.intrinsic_excitability[layer_key] = torch.clamp(self.intrinsic_excitability[layer_key], 0.1, 10.0)
        
        # Apply homeostatic scaling
        scaled_activations = activations * self.intrinsic_excitability[layer_key].unsqueeze(0)
        
        # Select neurons that best maintain homeostasis
        homeostatic_scores = self.intrinsic_excitability[layer_key] * current_activity
        selected_indices = torch.topk(homeostatic_scores, top_k).indices
        
        metadata = {
            'selection_type': 'homeostatic',
            'homeostatic_scores': homeostatic_scores,
            'intrinsic_excitability': self.intrinsic_excitability[layer_key],
            'activity_targets': self.activity_targets[layer_key],
            'current_activity': current_activity
        }
        
        return selected_indices, metadata
    
    def _evolutionary_selection(self, activations: torch.Tensor, targets: Optional[torch.Tensor],
                               top_k: int, layer_name: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Evolutionary selection based on fitness.
        
        Implements genetic algorithm principles for neuron selection.
        """
        num_neurons = activations.shape[-1]
        device = activations.device
        
        # Initialize fitness scores if needed
        layer_key = f"{layer_name}_fitness"
        if layer_key not in self.fitness_scores:
            self.fitness_scores[layer_key] = torch.ones(num_neurons, device=device)
        
        # Compute fitness based on activation patterns and targets
        if targets is not None:
            # Supervised fitness: correlation with targets
            fitness = torch.zeros(num_neurons, device=device)
            for i in range(num_neurons):
                neuron_acts = activations[:, i]
                if targets.dim() == 1:
                    target_vals = targets
                else:
                    target_vals = targets[:, i % targets.shape[1]]
                
                corr = torch.corrcoef(torch.stack([neuron_acts, target_vals]))[0, 1]
                fitness[i] = torch.abs(corr) if not torch.isnan(corr) else 0.0
        else:
            # Unsupervised fitness: activation magnitude and diversity
            magnitude_fitness = torch.mean(torch.abs(activations), dim=0)
            
            # Diversity bonus: neurons with unique activation patterns
            activation_std = torch.std(activations, dim=0)
            diversity_fitness = activation_std / (torch.mean(activation_std) + 1e-8)
            
            fitness = magnitude_fitness * diversity_fitness
        
        # Update fitness with exponential moving average
        self.fitness_scores[layer_key] = (self.config.decay_rate * self.fitness_scores[layer_key] + 
                                        (1 - self.config.decay_rate) * fitness)
        
        # Evolutionary selection with elitism and tournament
        elite_size = max(1, top_k // 4)  # Top 25% are elite
        tournament_size = min(3, num_neurons)
        
        # Elite selection
        elite_indices = torch.topk(self.fitness_scores[layer_key], elite_size).indices
        
        # Tournament selection for remaining slots
        remaining_slots = top_k - elite_size
        tournament_selected = []
        
        for _ in range(remaining_slots):
            # Random tournament
            tournament_indices = torch.randint(0, num_neurons, (tournament_size,), device=device)
            tournament_fitness = self.fitness_scores[layer_key][tournament_indices]
            winner_idx = tournament_indices[torch.argmax(tournament_fitness)]
            tournament_selected.append(winner_idx)
        
        selected_indices = torch.cat([elite_indices, torch.stack(tournament_selected)])
        
        self.generation_count += 1
        
        metadata = {
            'selection_type': 'evolutionary',
            'fitness_scores': self.fitness_scores[layer_key],
            'elite_indices': elite_indices,
            'generation': self.generation_count,
            'mean_fitness': torch.mean(self.fitness_scores[layer_key]).item()
        }
        
        return selected_indices, metadata
    
    def _stdp_selection(self, activations: torch.Tensor, top_k: int, 
                       layer_name: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Spike-Timing Dependent Plasticity (STDP) selection.
        
        Selects neurons based on temporal correlation patterns.
        """
        batch_size, num_neurons = activations.shape
        device = activations.device
        
        # Convert activations to spike times (simplified)
        spike_threshold = torch.mean(activations) + torch.std(activations)
        spike_mask = activations > spike_threshold
        
        # Update spike timing records
        layer_key = f"{layer_name}_spikes"
        if layer_key not in self.spike_times:
            self.spike_times[layer_key] = [[] for _ in range(num_neurons)]
        
        current_time = len(self.activation_history)
        for neuron_idx in range(num_neurons):
            if torch.any(spike_mask[:, neuron_idx]):
                self.spike_times[layer_key][neuron_idx].append(current_time)
                # Keep only recent spikes
                self.spike_times[layer_key][neuron_idx] = \
                    self.spike_times[layer_key][neuron_idx][-self.stdp_window:]
        
        # Compute STDP-based connectivity strength
        stdp_scores = torch.zeros(num_neurons, device=device)
        
        for i in range(num_neurons):
            for j in range(num_neurons):
                if i != j:
                    # Compute temporal correlation
                    spikes_i = self.spike_times[layer_key][i]
                    spikes_j = self.spike_times[layer_key][j]
                    
                    if spikes_i and spikes_j:
                        # STDP rule: potentiation if pre before post, depression otherwise
                        for spike_i in spikes_i:
                            for spike_j in spikes_j:
                                dt = spike_j - spike_i
                                if abs(dt) <= self.stdp_window:
                                    if dt > 0:  # Pre before post: potentiation
                                        stdp_scores[i] += torch.exp(torch.tensor(-abs(dt) / 10.0))
                                    else:  # Post before pre: depression
                                        stdp_scores[i] -= 0.5 * torch.exp(torch.tensor(-abs(dt) / 10.0))
        
        # Select neurons with highest STDP scores
        selected_indices = torch.topk(stdp_scores, top_k).indices
        
        metadata = {
            'selection_type': 'stdp',
            'stdp_scores': stdp_scores,
            'spike_counts': [len(spikes) for spikes in self.spike_times[layer_key]],
            'mean_stdp_score': torch.mean(stdp_scores).item()
        }
        
        return selected_indices, metadata
    
    def _criticality_selection(self, activations: torch.Tensor, top_k: int, 
                              layer_name: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Neural criticality-based selection.
        
        Selects neurons that maintain the network at the edge of chaos.
        """
        batch_size, num_neurons = activations.shape
        device = activations.device
        
        # Detect neural avalanches (cascades of activity)
        threshold = torch.mean(activations) + 0.5 * torch.std(activations)
        active_neurons = (activations > threshold).float()
        
        # Compute avalanche sizes
        avalanche_sizes = torch.sum(active_neurons, dim=-1)
        self.avalanche_sizes.extend(avalanche_sizes.cpu().tolist())
        
        # Estimate branching parameter (criticality measure)
        if len(self.avalanche_sizes) > 10:
            sizes = torch.tensor(list(self.avalanche_sizes))
            # Branching parameter approximation
            self.branching_parameter = torch.mean(sizes[1:] / (sizes[:-1] + 1e-8)).item()
        
        # Select neurons that contribute to maintaining criticality
        # Neurons that are active but not overly dominant
        activity_contribution = torch.mean(active_neurons, dim=0)
        
        # Criticality score: balance between activity and restraint
        target_criticality = self.config.criticality_target
        criticality_deviation = torch.abs(activity_contribution - target_criticality)
        criticality_scores = 1.0 / (1.0 + criticality_deviation)
        
        # Bonus for neurons that help maintain optimal branching parameter
        branching_target = 1.0  # Critical branching parameter
        branching_bonus = 1.0 / (1.0 + abs(self.branching_parameter - branching_target))
        criticality_scores *= branching_bonus
        
        selected_indices = torch.topk(criticality_scores, top_k).indices
        
        metadata = {
            'selection_type': 'criticality',
            'criticality_scores': criticality_scores,
            'branching_parameter': self.branching_parameter,
            'avalanche_sizes': list(self.avalanche_sizes)[-10:],  # Last 10
            'activity_contribution': activity_contribution
        }
        
        return selected_indices, metadata
    
    def _multi_objective_selection(self, activations: torch.Tensor, targets: Optional[torch.Tensor],
                                  weights: Optional[torch.Tensor], top_k: int, 
                                  layer_name: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Multi-objective selection combining multiple biological principles.
        
        Combines Hebbian learning, plasticity, competition, and homeostasis.
        """
        # Get scores from different methods
        hebbian_indices, hebbian_meta = self._hebbian_selection(activations, targets, top_k, layer_name)
        plasticity_indices, plasticity_meta = self._plasticity_selection(activations, None, None, top_k, layer_name)
        competitive_indices, competitive_meta = self._competitive_selection(activations, top_k, layer_name)
        homeostatic_indices, homeostatic_meta = self._homeostatic_selection(activations, top_k, layer_name)
        
        # Combine scores with weights
        num_neurons = activations.shape[-1]
        device = activations.device
        
        combined_scores = torch.zeros(num_neurons, device=device)
        
        # Add weighted contributions
        weights_dict = {
            'hebbian': 0.3,
            'plasticity': 0.25,
            'competitive': 0.25,
            'homeostatic': 0.2
        }
        
        # Normalize and combine scores
        hebbian_scores = hebbian_meta['hebbian_scores']
        hebbian_scores = (hebbian_scores - hebbian_scores.min()) / (hebbian_scores.max() - hebbian_scores.min() + 1e-8)
        
        plasticity_scores = plasticity_meta['plasticity_scores']
        plasticity_scores = (plasticity_scores - plasticity_scores.min()) / (plasticity_scores.max() - plasticity_scores.min() + 1e-8)
        
        competitive_scores = competitive_meta['competition_scores']
        competitive_scores = (competitive_scores - competitive_scores.min()) / (competitive_scores.max() - competitive_scores.min() + 1e-8)
        
        homeostatic_scores = homeostatic_meta['homeostatic_scores']
        homeostatic_scores = (homeostatic_scores - homeostatic_scores.min()) / (homeostatic_scores.max() - homeostatic_scores.min() + 1e-8)
        
        combined_scores = (weights_dict['hebbian'] * hebbian_scores +
                          weights_dict['plasticity'] * plasticity_scores +
                          weights_dict['competitive'] * competitive_scores +
                          weights_dict['homeostatic'] * homeostatic_scores)
        
        selected_indices = torch.topk(combined_scores, top_k).indices
        
        metadata = {
            'selection_type': 'multi_objective',
            'combined_scores': combined_scores,
            'component_scores': {
                'hebbian': hebbian_scores,
                'plasticity': plasticity_scores,
                'competitive': competitive_scores,
                'homeostatic': homeostatic_scores
            },
            'weights': weights_dict
        }
        
        return selected_indices, metadata
    
    def _compute_winner_frequency(self, num_neurons: int) -> torch.Tensor:
        """Compute frequency of each neuron being a winner."""
        if not self.winner_history:
            return torch.zeros(num_neurons)
        
        winner_counts = torch.zeros(num_neurons)
        for winner in self.winner_history:
            if winner < num_neurons:
                winner_counts[winner] += 1
        
        return winner_counts / len(self.winner_history)
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the selection process."""
        return {
            'selection_type': self.selection_type,
            'generation_count': self.generation_count,
            'correlation_matrix_shape': self.correlation_matrix.shape if self.correlation_matrix is not None else None,
            'plasticity_traces_keys': list(self.plasticity_traces.keys()),
            'fitness_scores_keys': list(self.fitness_scores.keys()),
            'activation_history_length': len(self.activation_history),
            'winner_history_length': len(self.winner_history),
            'avalanche_history_length': len(self.avalanche_sizes),
            'branching_parameter': getattr(self, 'branching_parameter', None)
        }
    
    def reset_state(self):
        """Reset all internal state for fresh learning."""
        self.correlation_matrix = None
        self.activation_history.clear()
        self.synaptic_strengths.clear()
        self.plasticity_traces.clear()
        self.weight_change_history.clear()
        self.inhibition_strengths.clear()
        self.winner_history.clear()
        self.activity_targets.clear()
        self.intrinsic_excitability.clear()
        self.homeostatic_scaling.clear()
        self.fitness_scores.clear()
        self.spike_times.clear()
        self.avalanche_sizes.clear()
        self.generation_count = 0
        self.branching_parameter = 1.0
        self.selection_history.clear()
        self.performance_metrics.clear()


def create_biological_selector(selection_type: str = 'hebbian', **kwargs) -> BiologicalNeuronSelector:
    """
    Factory function to create a biological neuron selector.
    
    Args:
        selection_type: Type of biological selection ('hebbian', 'plasticity', 'competitive', 
                       'homeostatic', 'evolutionary', 'stdp', 'criticality', 'multi_objective')
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured BiologicalNeuronSelector instance
    """
    config = BiologicalSelectionConfig(selection_type=selection_type, **kwargs)
    return BiologicalNeuronSelector(config)


# Example usage and integration with existing CTM code
def integrate_with_ctm_get_neuron_select_type(self, biological_selector: Optional[BiologicalNeuronSelector] = None):
    """
    Enhanced version of get_neuron_select_type that includes biological methods.
    
    This function can replace the existing get_neuron_select_type method in CTM.
    """
    print(f"Using neuron select type: {self.neuron_select_type}")
    
    # Handle biological selection types
    if self.neuron_select_type.startswith('bio_'):
        bio_type = self.neuron_select_type[4:]  # Remove 'bio_' prefix
        if biological_selector is None:
            biological_selector = create_biological_selector(bio_type)
        
        # For biological methods, we use the same type for both out and action
        neuron_select_type_out = neuron_select_type_action = self.neuron_select_type
        return neuron_select_type_out, neuron_select_type_action
    
    # Handle legacy types
    elif self.neuron_select_type == 'first-last':
        neuron_select_type_out, neuron_select_type_action = 'first', 'last'
    elif self.neuron_select_type in ('random', 'random-pairing'):
        neu