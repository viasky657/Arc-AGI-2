"""
Enhanced Neuron Selection for CTM Integration

This module provides an enhanced version of the CTM neuron selection system
that integrates biologically-inspired methods with the existing random approaches.
It's designed to be a drop-in replacement for the current get_neuron_select_type method.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from .biological_neuron_selection import BiologicalNeuronSelector, BiologicalSelectionConfig, create_biological_selector

# Extended valid neuron selection types
ENHANCED_NEURON_SELECT_TYPES = [
    # Legacy types
    'first-last', 'random', 'random-pairing',
    # Biologically-inspired types
    'bio_hebbian', 'bio_plasticity', 'bio_competitive', 'bio_homeostatic',
    'bio_evolutionary', 'bio_stdp', 'bio_criticality', 'bio_multi_objective',
    # Hybrid approaches
    'adaptive_random', 'performance_guided', 'task_aware'
]


class EnhancedNeuronSelector:
    """
    Enhanced neuron selector that combines biological and traditional methods.
    
    This class can be integrated into existing CTM models to provide more
    sophisticated neuron selection strategies while maintaining backward compatibility.
    """
    
    def __init__(self, neuron_select_type: str = 'random-pairing', **kwargs):
        self.neuron_select_type = neuron_select_type
        self.biological_selector = None
        self.performance_history = []
        self.adaptation_enabled = kwargs.get('adaptation_enabled', True)
        self.performance_threshold = kwargs.get('performance_threshold', 0.1)
        
        # Initialize biological selector if needed
        if self.neuron_select_type.startswith('bio_'):
            bio_type = self.neuron_select_type[4:]
            self.biological_selector = create_biological_selector(bio_type, **kwargs)
        
        # Task-specific selection strategies
        self.task_strategies = {
            'classification': 'bio_competitive',
            'regression': 'bio_hebbian', 
            'generation': 'bio_plasticity',
            'rl': 'bio_evolutionary',
            'sequence': 'bio_stdp'
        }
        
    def get_neuron_select_type(self):
        """
        Enhanced version of CTM's get_neuron_select_type method.
        
        Returns appropriate neuron selection types for output and action synchronization.
        """
        print(f"Using enhanced neuron select type: {self.neuron_select_type}")
        
        # Handle biological selection types
        if self.neuron_select_type.startswith('bio_'):
            # For biological methods, use the same type for both out and action
            neuron_select_type_out = neuron_select_type_action = self.neuron_select_type
            return neuron_select_type_out, neuron_select_type_action
        
        # Handle adaptive methods
        elif self.neuron_select_type == 'adaptive_random':
            # Start with random, adapt based on performance
            if len(self.performance_history) > 10:
                recent_performance = np.mean(self.performance_history[-10:])
                if recent_performance < self.performance_threshold:
                    # Switch to biological method if performance is poor
                    return 'bio_hebbian', 'bio_plasticity'
            return 'random', 'random-pairing'
        
        elif self.neuron_select_type == 'performance_guided':
            # Choose method based on recent performance trends
            if len(self.performance_history) > 5:
                trend = np.polyfit(range(5), self.performance_history[-5:], 1)[0]
                if trend < 0:  # Performance declining
                    return 'bio_competitive', 'bio_evolutionary'
                else:  # Performance stable/improving
                    return 'bio_hebbian', 'bio_plasticity'
            return 'random', 'random'
        
        elif self.neuron_select_type == 'task_aware':
            # This would need task information passed in
            # For now, default to multi-objective
            return 'bio_multi_objective', 'bio_multi_objective'
        
        # Handle legacy types
        elif self.neuron_select_type == 'first-last':
            neuron_select_type_out, neuron_select_type_action = 'first', 'last'
        elif self.neuron_select_type in ('random', 'random-pairing'):
            neuron_select_type_out = neuron_select_type_action = self.neuron_select_type
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
        
        return neuron_select_type_out, neuron_select_type_action
    
    def select_neurons_for_synchronization(self, 
                                         activations: torch.Tensor,
                                         synch_type: str,
                                         n_synch: int,
                                         d_model: int,
                                         targets: Optional[torch.Tensor] = None,
                                         weights: Optional[torch.Tensor] = None,
                                         **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced neuron selection for synchronization.
        
        Args:
            activations: Current neuron activations
            synch_type: Type of synchronization ('out' or 'action')
            n_synch: Number of neurons to select
            d_model: Model dimension
            targets: Optional target values for supervised selection
            weights: Optional weight matrices for plasticity-based selection
            
        Returns:
            Tuple of (left_neuron_indices, right_neuron_indices)
        """
        device = activations.device if activations is not None else torch.device('cpu')
        
        # Get selection types
        neuron_select_type_out, neuron_select_type_action = self.get_neuron_select_type()
        current_type = neuron_select_type_out if synch_type == 'out' else neuron_select_type_action
        
        # Handle biological selection
        if current_type.startswith('bio_') and self.biological_selector is not None:
            if activations is not None:
                selected_indices, metadata = self.biological_selector.select_neurons(
                    activations=activations,
                    targets=targets,
                    weights=weights,
                    top_k=n_synch,
                    layer_name=f"{synch_type}_synch"
                )
                
                # For synchronization, we need pairs of neurons
                if len(selected_indices) >= n_synch:
                    neuron_indices_left = selected_indices[:n_synch]
                    
                    # Create right indices based on selection strategy
                    bio_type = current_type[4:]
                    if bio_type in ['competitive', 'evolutionary']:
                        # Use different neurons for right side to encourage diversity
                        remaining_indices = torch.tensor([i for i in range(d_model) 
                                                        if i not in selected_indices[:n_synch]], device=device)
                        if len(remaining_indices) >= n_synch:
                            neuron_indices_right = remaining_indices[:n_synch]
                        else:
                            neuron_indices_right = selected_indices[:n_synch]  # Fallback
                    else:
                        # Use same neurons for both sides (self-synchronization)
                        neuron_indices_right = neuron_indices_left
                        
                else:
                    # Fallback to random if not enough neurons selected
                    neuron_indices_left = torch.from_numpy(
                        np.random.choice(np.arange(d_model), size=n_synch, replace=False)
                    ).to(device)
                    neuron_indices_right = neuron_indices_left
            else:
                # No activations available, fall back to random
                neuron_indices_left = torch.from_numpy(
                    np.random.choice(np.arange(d_model), size=n_synch, replace=False)
                ).to(device)
                neuron_indices_right = neuron_indices_left
        
        # Handle legacy selection types
        elif current_type == 'first':
            neuron_indices_left = torch.arange(n_synch, device=device)
            neuron_indices_right = neuron_indices_left
        elif current_type == 'last':
            neuron_indices_left = torch.arange(d_model - n_synch, d_model, device=device)
            neuron_indices_right = neuron_indices_left
        elif current_type == 'random':
            neuron_indices_left = torch.from_numpy(
                np.random.choice(np.arange(d_model), size=n_synch, replace=False)
            ).to(device)
            neuron_indices_right = torch.from_numpy(
                np.random.choice(np.arange(d_model), size=n_synch, replace=False)
            ).to(device)
        elif current_type == 'random-pairing':
            neuron_indices_left = torch.from_numpy(
                np.random.choice(np.arange(d_model), size=n_synch, replace=False)
            ).to(device)
            neuron_indices_right = torch.from_numpy(
                np.random.choice(np.arange(d_model), size=n_synch, replace=False)
            ).to(device)
        else:
            raise ValueError(f"Unknown neuron selection type: {current_type}")
        
        return neuron_indices_left, neuron_indices_right
    
    def update_performance(self, performance_metric: float):
        """Update performance history for adaptive selection."""
        self.performance_history.append(performance_metric)
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def adapt_selection_strategy(self, task_type: Optional[str] = None):
        """Adapt selection strategy based on task type or performance."""
        if task_type and task_type in self.task_strategies:
            new_strategy = self.task_strategies[task_type]
            if new_strategy != self.neuron_select_type:
                print(f"Adapting neuron selection from {self.neuron_select_type} to {new_strategy} for task {task_type}")
                self.neuron_select_type = new_strategy
                
                # Reinitialize biological selector if needed
                if new_strategy.startswith('bio_'):
                    bio_type = new_strategy[4:]
                    self.biological_selector = create_biological_selector(bio_type)
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics about the selection process."""
        stats = {
            'current_type': self.neuron_select_type,
            'performance_history_length': len(self.performance_history),
            'recent_performance': np.mean(self.performance_history[-10:]) if len(self.performance_history) >= 10 else None
        }
        
        if self.biological_selector:
            stats.update(self.biological_selector.get_selection_stats())
        
        return stats


def create_enhanced_ctm_with_biological_selection(original_ctm_class):
    """
    Factory function to create an enhanced CTM class with biological neuron selection.
    
    This function takes an existing CTM class and returns a new class with enhanced
    neuron selection capabilities.
    """
    
    class EnhancedCTM(original_ctm_class):
        def __init__(self, *args, **kwargs):
            # Extract biological selection parameters
            bio_selection_config = kwargs.pop('biological_selection_config', {})
            
            super().__init__(*args, **kwargs)
            
            # Replace the neuron selector
            self.enhanced_selector = EnhancedNeuronSelector(
                neuron_select_type=self.neuron_select_type,
                **bio_selection_config
            )
        
        def get_neuron_select_type(self):
            """Enhanced neuron selection method."""
            return self.enhanced_selector.get_neuron_select_type()
        
        def initialize_left_right_neurons(self, synch_type, d_model, n_synch, n_random_pairing_self=0):
            """Enhanced neuron initialization with biological selection."""
            # Get current activations if available
            activations = getattr(self, '_current_activations', None)
            targets = getattr(self, '_current_targets', None)
            weights = getattr(self, '_current_weights', None)
            
            neuron_indices_left, neuron_indices_right = self.enhanced_selector.select_neurons_for_synchronization(
                activations=activations,
                synch_type=synch_type,
                n_synch=n_synch,
                d_model=d_model,
                targets=targets,
                weights=weights
            )
            
            return neuron_indices_left, neuron_indices_right
        
        def forward(self, x, track=False, **kwargs):
            """Enhanced forward pass that provides context for biological selection."""
            # Store current state for biological selection
            if hasattr(x, 'shape') and len(x.shape) >= 2:
                self._current_activations = x
            
            # Store targets if provided
            if 'targets' in kwargs:
                self._current_targets = kwargs['targets']
            
            # Call original forward method
            result = super().forward(x, track=track, **kwargs)
            
            # Update performance if loss is available
            if hasattr(result, 'loss') or (isinstance(result, dict) and 'loss' in result):
                loss_value = result.loss if hasattr(result, 'loss') else result['loss']
                if torch.is_tensor(loss_value):
                    self.enhanced_selector.update_performance(-loss_value.item())  # Negative because lower loss is better
            
            return result
        
        def adapt_to_task(self, task_type: str):
            """Adapt neuron selection strategy to specific task type."""
            self.enhanced_selector.adapt_selection_strategy(task_type)
        
        def get_enhanced_selection_stats(self):
            """Get enhanced selection statistics."""
            return self.enhanced_selector.get_selection_stats()
    
    return EnhancedCTM


# Example integration with existing CTM
def integrate_biological_selection_into_ctm():
    """
    Example of how to integrate biological selection into existing CTM code.
    
    This shows how to modify the existing get_neuron_select_type method.
    """
    
    # This would replace the existing method in OriginalCTMCore
    def enhanced_get_neuron_select_type(self):
        """
        Enhanced version that supports biological selection methods.
        """
        # Create enhanced selector if not exists
        if not hasattr(self, '_enhanced_selector'):
            self._enhanced_selector = EnhancedNeuronSelector(
                neuron_select_type=self.neuron_select_type
            )
        
        return self._enhanced_selector.get_neuron_select_type()
    
    # This would replace the existing method in OriginalCTMCore  
    def enhanced_initialize_left_right_neurons(self, synch_type, d_model, n_synch, n_random_pairing_self=0):
        """
        Enhanced neuron initialization with biological selection.
        """
        if not hasattr(self, '_enhanced_selector'):
            self._enhanced_selector = EnhancedNeuronSelector(
                neuron_select_type=self.neuron_select_type
            )
        
        # Get current context for biological selection
        activations = getattr(self, '_last_hidden_state', None)
        
        return self._enhanced_selector.select_neurons_for_synchronization(
            activations=activations,
            synch_type=synch_type,
            n_synch=n_synch,
            d_model=d_model
        )
    
    return enhanced_get_neuron_select_type, enhanced_initialize_left_right_neurons


# Validation function
def validate_enhanced_selection():
    """Validate that enhanced selection works correctly."""
    
    # Test configuration
    config = BiologicalSelectionConfig(
        selection_type='hebbian',
        sparsity_ratio=0.5,
        learning_rate=0.01
    )
    
    # Create selector
    selector = EnhancedNeuronSelector('bio_hebbian')
    
    # Test with dummy data
    batch_size, num_neurons = 32, 512
    activations = torch.randn(batch_size, num_neurons)
    targets = torch.randn(batch_size, 10)
    
    # Test selection
    left_indices, right_indices = selector.select_neurons_for_synchronization(
        activations=activations,
        synch_type='out',
        n_synch=64,
        d_model=num_neurons,
        targets=targets
    )
    
    print(f"Selected {len(left_indices)} left neurons and {len(right_indices)} right neurons")
    print(f"Selection stats: {selector.get_selection_stats()}")
    
    return True


if __name__ == "__main__":
    # Run validation
    validate_enhanced_selection()
    print("Enhanced neuron selection validation completed successfully!")