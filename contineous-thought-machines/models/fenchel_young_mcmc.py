import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Callable, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math

from .modules import SuperLinear
from .utils import compute_normalized_entropy
from itertools import combinations

@dataclass
class MCMCConfig:
    """Configuration for MCMC sampling parameters"""
    num_chains: int = 5
    chain_length: int = 1000
    burn_in: int = 100
    temperature_schedule: str = "geometric"  # "geometric", "linear", "constant"
    initial_temp: float = 10.0
    final_temp: float = 1.0
    decay_rate: float = 0.995
    neighborhood_radius: int = 1
    initialization_method: str = "persistent"  # "random", "persistent", "data_based"

class TemperatureScheduler:
    """Temperature annealing schedules for MCMC sampling"""
    
    @staticmethod
    def geometric(initial_temp: float, decay_rate: float, final_temp: float):
        def schedule(step: int) -> float:
            return max(initial_temp * (decay_rate ** step), final_temp)
        return schedule
    
    @staticmethod
    def linear(initial_temp: float, final_temp: float, total_steps: int):
        def schedule(step: int) -> float:
            progress = min(step / total_steps, 1.0)
            return initial_temp * (1 - progress) + final_temp * progress
        return schedule
    
    @staticmethod
    def constant(temperature: float):
        def schedule(step: int) -> float:
            return temperature
        return schedule

class DiscreteOutputSpace:
    """Base class for discrete output spaces with neighborhood structures"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        # Defer generation of full output_space if it's too large or not needed immediately
        self._full_output_space_generated = False
        try:
            # Attempt to generate if dimension is small, for random_state compatibility
            if self.dimension <= 10: # Arbitrary small threshold
                 self.output_space = self._generate_space()
                 self._full_output_space_generated = True
            else:
                 self.output_space = [] # Placeholder
        except NotImplementedError:
            self.output_space = []
        except ValueError: # Handles cases like BinaryHypercube dimension > 20
            self.output_space = []

    def _generate_space(self) -> List[torch.Tensor]:
        """
        Generate the discrete output space. 
        Called cautiously during __init__ only if dimension is small.
        Subclasses should implement this.
        """
        raise NotImplementedError

    def get_available_neighborhood_strategies(self, state: Optional[torch.Tensor] = None) -> List[str]:
        """
        Returns a list of names for different ways to get neighbors.
        Subclasses should override this.
        'state' is optional, might be needed for state-dependent strategies.
        Example: ['radius_1', 'radius_2'] for BinaryHypercube
        """
        raise NotImplementedError

    def get_neighbors(self, state: torch.Tensor, strategy_name: str, **strategy_params) -> List[torch.Tensor]:
        """
        Get neighbors of a state using a specific strategy.
        Subclasses should override this.
        strategy_params could include radius, num_swaps etc.
        """
        raise NotImplementedError

    def get_proposal_prob(self, current_state: torch.Tensor, proposed_state: torch.Tensor, strategy_name: str, **strategy_params) -> float:
        """
        Compute proposal probability q_s(proposed_state | current_state) for a given strategy.
        This is typically 1/|N_s(current_state)| if proposed_state is in N_s(current_state), and 0 otherwise,
        assuming a uniform proposal over the neighborhood N_s.
        """
        # Ensure strategy_params are passed correctly to get_neighbors
        neighbors = self.get_neighbors(current_state, strategy_name, **strategy_params)
        if not neighbors: # Or if proposed_state cannot be formed from current_state via strategy
            return 0.0
        
        is_neighbor = any(torch.allclose(neighbor, proposed_state) for neighbor in neighbors)
        
        if is_neighbor:
            return 1.0 / len(neighbors)
        return 0.0

    def random_state(self) -> torch.Tensor:
        """Sample a random state from the output space"""
        if self._full_output_space_generated and self.output_space:
            return random.choice(self.output_space).clone()
        
        # Fallback for large spaces: try to generate one efficiently
        single_random_member = self._generate_random_member_directly()
        if single_random_member is not None:
            return single_random_member
        
        # If still no state, and full space wasn't generated due to size, this is an issue.
        # As a last resort, if output_space is empty but _full_output_space_generated is False,
        # it means we deferred generation. Try generating now if it wasn't due to an error.
        if not self.output_space and not self._full_output_space_generated:
            try:
                # This call might be expensive or raise ValueError if dim is too large
                self.output_space = self._generate_space()
                self._full_output_space_generated = True # Mark that we tried
                if self.output_space:
                    return random.choice(self.output_space).clone()
            except (NotImplementedError, ValueError) as e:
                # If generation fails here, we truly cannot produce a random state this way.
                raise RuntimeError(
                    f"Cannot generate random_state for {self.__class__.__name__} with dimension {self.dimension}. "
                    f"Full space generation failed or not implemented for large dimensions. Error: {e}. "
                    "Consider implementing _generate_random_member_directly()."
                )
        
        if not self.output_space: # If still no output space after all attempts
            raise RuntimeError(
                f"Cannot generate random_state for {self.__class__.__name__} with dimension {self.dimension}. "
                "Output space is empty and _generate_random_member_directly() did not provide a state."
            )
        return random.choice(self.output_space).clone() # Should only be reached if space was generated.

    def _generate_random_member_directly(self) -> Optional[torch.Tensor]:
        """
        Subclasses can override this for efficient random state generation 
        without full enumeration of the entire space.
        """
        return None

class BinaryHypercube(DiscreteOutputSpace):
    """Binary hypercube {0,1}^d with Hamming distance neighborhoods"""
    
    def _generate_space(self) -> List[torch.Tensor]:
        """Generate all binary vectors of given dimension"""
        if self.dimension > 16:  # Adjusted threshold for practical enumeration
            # print(f"Warning: Dimension {self.dimension} for BinaryHypercube is large for explicit enumeration. Space not pre-generated.")
            return [] # Do not generate by default if too large
        
        return [torch.tensor([int(b) for b in format(i, f'0{self.dimension}b')], 
                           dtype=torch.float32) 
                for i in range(2**self.dimension)]

    def get_available_neighborhood_strategies(self, state: Optional[torch.Tensor] = None) -> List[str]:
        """Returns available Hamming distance based strategies."""
        # Example: allow flipping 1, 2, or 3 bits if dimension is large enough
        strategies = ["radius_1"]
        if self.dimension >= 2:
            strategies.append("radius_2")
        if self.dimension >= 3:
            strategies.append("radius_3")
        return strategies

    def get_neighbors(self, state: torch.Tensor, strategy_name: str, **strategy_params) -> List[torch.Tensor]:
        """Find neighbors within Hamming distance `radius`."""
        neighbors = []
        state_np = state.numpy().astype(int) # Ensure state is usable as numpy int array

        if not strategy_name.startswith("radius_"):
            raise ValueError(f"Unknown strategy for BinaryHypercube: {strategy_name}. Expected 'radius_N'.")
        
        try:
            radius = int(strategy_name.split("_")[1])
        except (IndexError, ValueError):
            # Fallback if strategy_params has radius (e.g. from old CorrectionRatioMCMC call)
            radius = strategy_params.get('radius', 1) 
            # print(f"Warning: Could not parse radius from strategy_name '{strategy_name}'. Defaulting to radius {radius} from params or 1.")


        if radius <= 0:
            return []
        
        # Optimized for radius=1
        if radius == 1:
            for i in range(self.dimension):
                neighbor_np = state_np.copy()
                neighbor_np[i] = 1 - neighbor_np[i]
                neighbors.append(torch.tensor(neighbor_np, dtype=torch.float32))
        else:
            # For larger radius, use recursive approach (can be expensive)
            # We need to collect all unique neighbors within the exact Hamming distance `radius`.
            # The previous _get_neighbors_recursive collected up to `radius`.
            # For exact radius, we can adapt or use a different approach for larger N.
            # For now, let's use a simplified version that finds all states at exact Hamming distance.
            
            # This is combinatorially explosive. For MCMC, typically small radii are used.
            # If larger radii are needed, a more efficient sampling of neighbors might be better
            # than enumerating all of them.
            # For now, let's keep the recursive one but ensure it finds exact radius.
            
            indices_to_flip_options = list(combinations(range(self.dimension), radius))
            for indices_to_flip in indices_to_flip_options:
                neighbor_np = state_np.copy()
                for idx in indices_to_flip:
                    neighbor_np[idx] = 1 - neighbor_np[idx]
                neighbors.append(torch.tensor(neighbor_np, dtype=torch.float32))

        return neighbors

    def _generate_random_member_directly(self) -> Optional[torch.Tensor]:
        """Generates a random binary vector of the given dimension."""
        if self.dimension <= 0:
            return None
        return torch.randint(0, 2, (self.dimension,), dtype=torch.float32)

class TopKPolytope(DiscreteOutputSpace):
    """Top-k polytope: binary vectors with exactly k ones"""
    
    def __init__(self, dimension: int, k: int):
        self.k = k
        if k < 0 or k > dimension: # k must be non-negative
            raise ValueError(f"k={k} must be between 0 and dimension={dimension}")
        super().__init__(dimension)
    
    def _generate_space(self) -> List[torch.Tensor]:
        """Generate all binary vectors with exactly k ones"""
        from itertools import combinations
        
        # Calculate binomial coefficient to estimate size
        if self.dimension > 0 and self.k > 0 and self.k <= self.dimension:
            try:
                num_combinations = math.comb(self.dimension, self.k)
                if num_combinations > 2**16: # Approx 65536, adjust as needed
                    # print(f"Warning: TopKPolytope space size ({num_combinations}) is large. Space not pre-generated.")
                    return []
            except AttributeError: # math.comb not available (e.g. older Python)
                 pass # Proceed with generation if not too large based on dim
        elif self.k == 0: # Only one vector (all zeros)
            return [torch.zeros(self.dimension, dtype=torch.float32)]


        if self.dimension > 20 and self.k > 0: # Heuristic to avoid large combinations
             # print(f"Warning: Dimension {self.dimension} for TopKPolytope is potentially large for explicit enumeration. Space not pre-generated.")
            return []

        space = []
        for positions in combinations(range(self.dimension), self.k):
            vector = torch.zeros(self.dimension, dtype=torch.float32)
            vector[list(positions)] = 1.0
            space.append(vector)
        
        return space

    def get_available_neighborhood_strategies(self, state: Optional[torch.Tensor] = None) -> List[str]:
        """Returns available swap-based strategies."""
        strategies = ["swaps_1"] # Always possible if k < dim and k > 0
        # Max possible swaps is min(k, dimension - k)
        max_swaps = min(self.k, self.dimension - self.k)
        if max_swaps >= 2:
            strategies.append("swaps_2")
        if max_swaps >= 3: # Example, can add more
            strategies.append("swaps_3")
        return strategies

    def get_neighbors(self, state: torch.Tensor, strategy_name: str, **strategy_params) -> List[torch.Tensor]:
        """Get neighbors by swapping `num_swaps` pairs of 0s and 1s"""
        from itertools import combinations # Moved import here
        
        neighbors = []
        state_np = state.numpy().astype(int)
        
        if not strategy_name.startswith("swaps_"):
            raise ValueError(f"Unknown strategy for TopKPolytope: {strategy_name}. Expected 'swaps_N'.")

        try:
            num_swaps = int(strategy_name.split("_")[1])
        except (IndexError, ValueError):
            num_swaps = strategy_params.get('num_swaps', 1) # Fallback
            # print(f"Warning: Could not parse num_swaps from strategy_name '{strategy_name}'. Defaulting to {num_swaps} from params or 1.")

        if num_swaps <= 0:
            return []

        ones_indices = np.where(state_np == 1)[0]
        zeros_indices = np.where(state_np == 0)[0]

        if len(ones_indices) < num_swaps or len(zeros_indices) < num_swaps:
            return [] # Not enough 1s or 0s to perform the required number of swaps

        for ones_to_swap_indices in combinations(ones_indices, num_swaps):
            for zeros_to_swap_indices in combinations(zeros_indices, num_swaps):
                neighbor_np = state_np.copy()
                neighbor_np[list(ones_to_swap_indices)] = 0
                neighbor_np[list(zeros_to_swap_indices)] = 1
                neighbors.append(torch.tensor(neighbor_np, dtype=torch.float32))
        
        return neighbors

    def _generate_random_member_directly(self) -> Optional[torch.Tensor]:
        """Generates a random binary vector with exactly k ones."""
        if self.dimension <= 0 and self.k > 0 :
            return None
        if self.k < 0 or self.k > self.dimension: # Should be caught by __init__
             return None

        vector = torch.zeros(self.dimension, dtype=torch.float32)
        if self.k > 0: # Only try to sample if k > 0
            chosen_indices = np.random.choice(self.dimension, self.k, replace=False)
            vector[chosen_indices] = 1.0
        return vector


class FenchelYoungMCMC(nn.Module):
    """
    Fenchel-Young loss with MCMC gradient estimation
    Integrates with CTM architecture for structured prediction
    """
    
    def __init__(self, 
                 output_space: DiscreteOutputSpace,
                 config: MCMCConfig,
                 phi_network: Optional[nn.Module] = None):
        super().__init__()
        
        self.output_space = output_space
        self.config = config
        self.phi_network = phi_network  # Optional structure function φ(y)
        
        # Initialize temperature scheduler
        self.temp_scheduler = self._create_temperature_scheduler()
        
        # Persistent chain states for warm starts
        self.persistent_states = None
        self.step_count = 0
        
        # Statistics tracking
        self.acceptance_rates = []
        self.chain_entropies = []
    
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
        else:  # constant
            return TemperatureScheduler.constant(self.config.final_temp)
    
    def phi_function(self, state: torch.Tensor) -> torch.Tensor:
        """Structure function φ(y) - can be learned or hand-crafted"""
        if self.phi_network is not None:
            return self.phi_network(state)
        return torch.tensor(0.0)  # Default: no structure bias
    
    def proposal_probability(self, from_state: torch.Tensor, to_state: torch.Tensor) -> float:
        """Compute proposal probability q(y'|y)"""
        neighbors = self.output_space.get_neighbors(from_state, self.config.neighborhood_radius)
        
        # Uniform proposal over neighbors
        for neighbor in neighbors:
            if torch.allclose(neighbor, to_state):
                return 1.0 / len(neighbors)
        
        return 0.0  # Not a neighbor
    
    def acceptance_ratio(self, 
                        current: torch.Tensor, 
                        proposal: torch.Tensor, 
                        theta: torch.Tensor, 
                        temperature: float) -> float:
        """Compute Metropolis-Hastings acceptance ratio"""
        # Energy difference
        current_energy = torch.dot(theta, current) + self.phi_function(current)
        proposal_energy = torch.dot(theta, proposal) + self.phi_function(proposal)
        energy_diff = proposal_energy - current_energy
        
        # Proposal probability ratio (correction for asymmetric proposals)
        q_ratio = self.proposal_probability(proposal, current) / \
                  max(self.proposal_probability(current, proposal), 1e-10)
        
        # Acceptance probability
        return float(q_ratio * torch.exp(energy_diff / temperature))
    
    def sample_chain(self, 
                    theta: torch.Tensor, 
                    chain_id: int = 0,
                    target_state: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """Run single MCMC chain"""
        # Initialize chain
        if self.config.initialization_method == "persistent" and self.persistent_states is not None:
            current_state = self.persistent_states[chain_id].clone()
        elif self.config.initialization_method == "data_based" and target_state is not None:
            current_state = target_state.clone()
        else:
            current_state = self.output_space.random_state()
        
        samples = []
        acceptances = 0
        total_steps = self.config.chain_length + self.config.burn_in
        
        for step in range(total_steps):
            temperature = self.temp_scheduler(step)
            
            # Propose new state
            neighbors = self.output_space.get_neighbors(current_state, self.config.neighborhood_radius)
            if not neighbors:
                continue  # Skip if no neighbors (shouldn't happen)
            
            proposal = random.choice(neighbors)
            
            # Metropolis-Hastings acceptance
            alpha = min(1.0, self.acceptance_ratio(current_state, proposal, theta, temperature))
            
            if random.random() < alpha:
                current_state = proposal
                acceptances += 1
            
            # Collect samples after burn-in
            if step >= self.config.burn_in:
                samples.append(current_state.clone())
        
        # Update persistent state
        if self.persistent_states is None:
            self.persistent_states = [None] * self.config.num_chains
        self.persistent_states[chain_id] = current_state.clone()
        
        # Compute statistics
        stats = {
            'acceptance_rate': acceptances / total_steps,
            'final_temperature': self.temp_scheduler(total_steps - 1),
            'chain_length': len(samples)
        }
        
        return samples, stats
    
    def estimate_expectation(self, 
                           theta: torch.Tensor,
                           target_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Estimate E[Y] under Gibbs distribution using multiple MCMC chains"""
        all_samples = []
        all_stats = []
        
        # Run multiple parallel chains
        for chain_id in range(self.config.num_chains):
            samples, stats = self.sample_chain(theta, chain_id, target_state)
            all_samples.extend(samples)
            all_stats.append(stats)
        
        # Estimate expectation
        if not all_samples:
            return torch.zeros_like(theta), {'error': 'No samples collected'}
        
        expectation = torch.mean(torch.stack(all_samples), dim=0)
        
        # Aggregate statistics
        avg_acceptance = np.mean([s['acceptance_rate'] for s in all_stats])
        sample_entropy = compute_normalized_entropy(torch.stack(all_samples))
        
        combined_stats = {
            'num_samples': len(all_samples),
            'avg_acceptance_rate': avg_acceptance,
            'sample_entropy': float(sample_entropy),
            'effective_sample_size': len(all_samples) / self.config.num_chains,
            'chain_stats': all_stats
        }
        
        return expectation, combined_stats
    
    def compute_fenchel_young_loss(self, 
                                  theta: torch.Tensor, 
                                  target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Compute Fenchel-Young loss and gradient"""
        # Estimate expectation E[Y] under π_θ,t
        expectation, stats = self.estimate_expectation(theta, target)
        
        # Fenchel-Young gradient: ∇_θ ℓ_t(θ; y) = E[Y] - y
        gradient = expectation - target
        
        # Approximate loss (up to constants)
        # ℓ_t(θ; y) = A_t(θ) + Ω_t(y) - ⟨θ, y⟩
        # We compute the differentiable part: ⟨θ, E[Y]⟩ - ⟨θ, y⟩
        loss = torch.dot(theta, expectation) - torch.dot(theta, target)
        
        return loss, gradient, stats
    
    def forward(self, 
               theta: torch.Tensor, 
               target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass: compute loss and return statistics"""
        loss, gradient, stats = self.compute_fenchel_young_loss(theta, target)
        
        # Store gradient for backward pass
        self._cached_gradient = gradient
        self.step_count += 1
        
        # Update tracking statistics
        if 'avg_acceptance_rate' in stats:
            self.acceptance_rates.append(stats['avg_acceptance_rate'])
        if 'sample_entropy' in stats:
            self.chain_entropies.append(stats['sample_entropy'])
        
        return loss, stats
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get MCMC diagnostics for monitoring"""
        return {
            'step_count': self.step_count,
            'recent_acceptance_rate': np.mean(self.acceptance_rates[-10:]) if self.acceptance_rates else 0.0,
            'recent_entropy': np.mean(self.chain_entropies[-10:]) if self.chain_entropies else 0.0,
            'config': self.config,
            'persistent_states_initialized': self.persistent_states is not None
        }

class CTMFenchelYoungIntegration(nn.Module):
    """
    Integration of Fenchel-Young MCMC with CTM architecture
    Enables structured prediction with continuous thought processes
    """
    
    def __init__(self,
                 input_dim: int,
                 output_space: DiscreteOutputSpace,
                 mcmc_config: MCMCConfig,
                 hidden_dim: int = 256,
                 num_thought_steps: int = 5):
        super().__init__()
        
        # CTM-style thinking network
        self.thought_network = nn.Sequential(
            SuperLinear(input_dim, hidden_dim),
            nn.ReLU(),
            SuperLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            SuperLinear(hidden_dim, output_space.dimension)
        )
        
        # Structure function network (optional φ(y))
        self.phi_network = nn.Sequential(
            nn.Linear(output_space.dimension, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Fenchel-Young MCMC module
        self.fy_mcmc = FenchelYoungMCMC(
            output_space=output_space,
            config=mcmc_config,
            phi_network=self.phi_network
        )
        
        self.num_thought_steps = num_thought_steps
        self.output_space = output_space
    
    def think(self, x: torch.Tensor, num_steps: Optional[int] = None) -> List[torch.Tensor]:
        """CTM-style iterative thinking process"""
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
               return_thoughts: bool = False) -> Dict[str, Any]:
        """
        Forward pass with thinking and structured prediction
        
        Args:
            x: Input features
            target: Target discrete structure
            return_thoughts: Whether to return intermediate thoughts
            
        Returns:
            Dictionary with loss, statistics, and optional thoughts
        """
        # Generate thoughts
        thoughts = self.think(x)
        final_theta = thoughts[-1]  # Use final thought as parameters
        
        # Compute Fenchel-Young loss using MCMC
        loss, mcmc_stats = self.fy_mcmc(final_theta, target)
        
        result = {
            'loss': loss,
            'mcmc_stats': mcmc_stats,
            'final_theta': final_theta
        }
        
        if return_thoughts:
            result['thoughts'] = thoughts
        
        return result
    
    def predict(self, x: torch.Tensor, return_distribution: bool = False) -> Dict[str, Any]:
        """Make prediction by sampling from learned distribution"""
        thoughts = self.think(x)
        final_theta = thoughts[-1]
        
        # Sample from the distribution
        expectation, stats = self.fy_mcmc.estimate_expectation(final_theta)
        
        # Find closest discrete structure
        min_dist = float('inf')
        best_structure = None
        
        for structure in self.output_space.output_space:
            dist = torch.norm(expectation - structure)
            if dist < min_dist:
                min_dist = dist
                best_structure = structure
        
        result = {
            'prediction': best_structure,
            'expectation': expectation,
            'confidence': 1.0 / (1.0 + min_dist),  # Simple confidence measure
            'stats': stats
        }
        
        if return_distribution:
            # Could implement full distribution estimation here
            result['distribution_samples'] = stats.get('samples', [])
        
        return result

# Example usage and testing
if __name__ == "__main__":
    # Test with binary classification
    print("Testing Fenchel-Young MCMC with binary hypercube...")
    
    # Setup
    dim = 5
    output_space = BinaryHypercube(dim)
    config = MCMCConfig(num_chains=3, chain_length=500)
    
    # Create integrated model
    model = CTMFenchelYoungIntegration(
        input_dim=10,
        output_space=output_space,
        mcmc_config=config,
        hidden_dim=64,
        num_thought_steps=3
    )
    
    # Test forward pass
    x = torch.randn(10)
    target = torch.randint(0, 2, (dim,)).float()
    
    result = model(x, target, return_thoughts=True)
    
    print(f"Loss: {result['loss']:.4f}")
    print(f"MCMC acceptance rate: {result['mcmc_stats']['avg_acceptance_rate']:.3f}")
    print(f"Number of thoughts: {len(result['thoughts'])}")
    
    # Test prediction
    pred_result = model.predict(x)
    print(f"Prediction: {pred_result['prediction']}")
    print(f"Confidence: {pred_result['confidence']:.3f}")