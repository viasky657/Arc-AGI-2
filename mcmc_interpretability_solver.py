"""
Black Box Solver for MCMC Layers Interpretability

This module provides tools to interpret and visualize the reasoning process
of Enhanced MCMC layers, extracting thought chains and decision pathways.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path
import warnings

@dataclass
class ThoughtStep:
    """Represents a single step in the model's thought process"""
    step_id: int
    layer_name: str
    input_state: torch.Tensor
    output_state: torch.Tensor
    attention_weights: Optional[torch.Tensor]
    mcmc_samples: Optional[torch.Tensor]
    confidence_score: float
    reasoning_vector: torch.Tensor
    energy_landscape: Dict[str, float]
    correction_ratio: Optional[float]
    metadata: Dict[str, Any]

@dataclass
class ReasoningChain:
    """Complete reasoning chain for a single forward pass"""
    input_data: torch.Tensor
    thought_steps: List[ThoughtStep]
    final_output: torch.Tensor
    confidence_trajectory: List[float]
    decision_points: List[int]
    reasoning_summary: str
    convergence_metrics: Dict[str, float]
    solver_diagnostics: List[Dict[str, Any]]

class MCMCInterpretabilityHook:
    """Hook to capture intermediate states during forward pass"""
    
    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.activations = []
        self.gradients = []
        self.attention_maps = []
        self.mcmc_states = []
        self.energy_values = []
        self.correction_ratios = []
        
    def forward_hook(self, module, input, output):
        """Capture forward pass information"""
        input_tensor = input[0] if isinstance(input, tuple) else input
        
        self.activations.append({
            'input': input_tensor.detach().clone(),
            'output': output.detach().clone(),
            'layer': self.layer_name,
            'timestamp': len(self.activations)
        })
        
        # Capture MCMC-specific information if available
        if hasattr(module, 'mcmc_samples'):
            self.mcmc_states.append(module.mcmc_samples.detach().clone())
        if hasattr(module, 'attention_weights'):
            self.attention_maps.append(module.attention_weights.detach().clone())
        if hasattr(module, 'correction_ratios') and module.correction_ratios:
            self.correction_ratios.append(module.correction_ratios[-1])
        if hasattr(module, 'solver_diagnostics') and module.solver_diagnostics:
            # Store energy landscape information
            diagnostics = module.solver_diagnostics[-1]
            if 'last_objective_value' in diagnostics:
                self.energy_values.append(diagnostics['last_objective_value'])
    
    def backward_hook(self, module, grad_input, grad_output):
        """Capture gradient information"""
        if grad_output and grad_output[0] is not None:
            self.gradients.append(grad_output[0].detach().clone())

class BlackBoxSolver:
    """Main interpretability solver for MCMC layers"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.hooks = {}
        self.reasoning_chains = []
        self.interpretation_cache = {}
        
        # Register hooks for all MCMC layers
        self._register_hooks()
        
    def _register_hooks(self):
        """Register hooks for all relevant layers"""
        for name, module in self.model.named_modules():
            if any(keyword in name.lower() for keyword in ['mcmc', 'enhanced', 'correction', 'fenchel']):
                hook = MCMCInterpretabilityHook(name)
                self.hooks[name] = hook
                
                # Register forward and backward hooks
                module.register_forward_hook(hook.forward_hook)
                module.register_backward_hook(hook.backward_hook)
    
    def extract_reasoning_chain(self, input_data: torch.Tensor, 
                              target: Optional[torch.Tensor] = None,
                              return_intermediate: bool = True) -> ReasoningChain:
        """Extract complete reasoning chain for given input"""
        
        # Clear previous captures
        for hook in self.hooks.values():
            hook.activations.clear()
            hook.gradients.clear()
            hook.attention_maps.clear()
            hook.mcmc_states.clear()
            hook.energy_values.clear()
            hook.correction_ratios.clear()
        
        # Forward pass with gradient computation
        self.model.eval()
        input_data = input_data.to(self.device)
        input_data.requires_grad_(True)
        
        with torch.enable_grad():
            if hasattr(self.model, 'forward') and 'return_thoughts' in self.model.forward.__code__.co_varnames:
                # Enhanced CTM model
                output = self.model.forward(input_data, target, return_thoughts=return_intermediate, return_diagnostics=True)
                final_output = output.get('loss', output) if isinstance(output, dict) else output
            else:
                # Standard model
                output = self.model(input_data)
                final_output = output
            
            # Backward pass if target provided
            if target is not None:
                if isinstance(final_output, dict) and 'loss' in final_output:
                    loss = final_output['loss']
                else:
                    loss = nn.functional.mse_loss(final_output, target.to(self.device))
                loss.backward()
        
        # Extract thought steps
        thought_steps = self._build_thought_steps()
        
        # Analyze confidence trajectory
        confidence_trajectory = self._compute_confidence_trajectory(thought_steps)
        
        # Identify decision points
        decision_points = self._identify_decision_points(confidence_trajectory)
        
        # Generate reasoning summary
        reasoning_summary = self._generate_reasoning_summary(thought_steps, output)
        
        # Compute convergence metrics
        convergence_metrics = self._compute_convergence_metrics(thought_steps)
        
        # Extract solver diagnostics
        solver_diagnostics = self._extract_solver_diagnostics()
        
        return ReasoningChain(
            input_data=input_data.detach(),
            thought_steps=thought_steps,
            final_output=final_output,
            confidence_trajectory=confidence_trajectory,
            decision_points=decision_points,
            reasoning_summary=reasoning_summary,
            convergence_metrics=convergence_metrics,
            solver_diagnostics=solver_diagnostics
        )
    
    def _build_thought_steps(self) -> List[ThoughtStep]:
        """Build thought steps from captured activations"""
        thought_steps = []
        
        for step_id, (layer_name, hook) in enumerate(self.hooks.items()):
            if not hook.activations:
                continue
                
            activation = hook.activations[-1]  # Get latest activation
            
            # Compute reasoning vector (dimensionality reduction of hidden state)
            reasoning_vector = self._compute_reasoning_vector(activation['output'])
            
            # Compute confidence score
            confidence_score = self._compute_confidence_score(activation['output'])
            
            # Compute energy landscape
            energy_landscape = self._compute_energy_landscape(activation, hook)
            
            # Get attention weights if available
            attention_weights = hook.attention_maps[-1] if hook.attention_maps else None
            
            # Get MCMC samples if available
            mcmc_samples = hook.mcmc_states[-1] if hook.mcmc_states else None
            
            # Get correction ratio if available
            correction_ratio = hook.correction_ratios[-1] if hook.correction_ratios else None
            
            thought_step = ThoughtStep(
                step_id=step_id,
                layer_name=layer_name,
                input_state=activation['input'],
                output_state=activation['output'],
                attention_weights=attention_weights,
                mcmc_samples=mcmc_samples,
                confidence_score=confidence_score,
                reasoning_vector=reasoning_vector,
                energy_landscape=energy_landscape,
                correction_ratio=correction_ratio,
                metadata={
                    'layer_type': type(self.model.get_submodule(layer_name)).__name__,
                    'activation_norm': torch.norm(activation['output']).item(),
                    'gradient_norm': torch.norm(hook.gradients[-1]).item() if hook.gradients else 0.0,
                    'timestamp': activation['timestamp']
                }
            )
            
            thought_steps.append(thought_step)
        
        return thought_steps
    
    def _compute_reasoning_vector(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Compute a low-dimensional reasoning vector from hidden state"""
        # Flatten if multi-dimensional
        if len(hidden_state.shape) > 2:
            hidden_state = hidden_state.flatten(1)
        
        # Compute top-k most important dimensions
        importance = torch.var(hidden_state, dim=0) if hidden_state.dim() > 1 else torch.abs(hidden_state)
        top_k = min(16, hidden_state.size(-1))
        _, top_indices = torch.topk(importance, top_k)
        
        # Extract reasoning vector
        if hidden_state.dim() > 1:
            reasoning_vector = hidden_state[:, top_indices].mean(0)
        else:
            reasoning_vector = hidden_state[top_indices]
        
        return reasoning_vector
    
    def _compute_confidence_score(self, hidden_state: torch.Tensor) -> float:
        """Compute confidence score based on activation patterns"""
        # Use entropy-based confidence measure
        if hidden_state.dim() > 1:
            # For multi-dimensional states, compute entropy across features
            probs = torch.softmax(hidden_state.flatten(), dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            max_entropy = torch.log(torch.tensor(len(probs), dtype=torch.float))
            confidence = 1.0 - (entropy / max_entropy).item()
        else:
            # For 1D states, use variance-based confidence
            confidence = torch.sigmoid(torch.var(hidden_state)).item()
        
        return confidence
    
    def _compute_energy_landscape(self, activation: Dict, hook: MCMCInterpretabilityHook) -> Dict[str, float]:
        """Compute energy landscape information"""
        energy_landscape = {}
        
        # Current energy
        if hook.energy_values:
            energy_landscape['current_energy'] = hook.energy_values[-1]
        
        # Energy gradient (if gradients available)
        if hook.gradients:
            grad_norm = torch.norm(hook.gradients[-1]).item()
            energy_landscape['gradient_norm'] = grad_norm
            energy_landscape['gradient_direction'] = 'ascending' if grad_norm > 0.1 else 'stable'
        
        # Activation-based energy proxy
        output_energy = torch.sum(activation['output'] ** 2).item()
        energy_landscape['activation_energy'] = output_energy
        
        return energy_landscape
    
    def _compute_confidence_trajectory(self, thought_steps: List[ThoughtStep]) -> List[float]:
        """Compute confidence trajectory across thought steps"""
        return [step.confidence_score for step in thought_steps]
    
    def _identify_decision_points(self, confidence_trajectory: List[float]) -> List[int]:
        """Identify key decision points in the reasoning process"""
        if len(confidence_trajectory) < 2:
            return []
        
        decision_points = []
        
        # Find points with significant confidence changes
        for i in range(1, len(confidence_trajectory)):
            confidence_change = abs(confidence_trajectory[i] - confidence_trajectory[i-1])
            if confidence_change > 0.1:  # Threshold for significant change
                decision_points.append(i)
        
        # Always include the final step
        if len(confidence_trajectory) - 1 not in decision_points:
            decision_points.append(len(confidence_trajectory) - 1)
        
        return decision_points
    
    def _generate_reasoning_summary(self, thought_steps: List[ThoughtStep], output: Any) -> str:
        """Generate human-readable reasoning summary"""
        if not thought_steps:
            return "No reasoning steps captured."
        
        summary_parts = []
        summary_parts.append(f"Reasoning chain with {len(thought_steps)} steps:")
        
        for i, step in enumerate(thought_steps):
            confidence_desc = "high" if step.confidence_score > 0.7 else "medium" if step.confidence_score > 0.4 else "low"
            
            step_desc = f"Step {i+1} ({step.layer_name}): {confidence_desc} confidence ({step.confidence_score:.3f})"
            
            if step.correction_ratio is not None:
                step_desc += f", correction ratio: {step.correction_ratio:.3f}"
            
            if 'current_energy' in step.energy_landscape:
                step_desc += f", energy: {step.energy_landscape['current_energy']:.3f}"
            
            summary_parts.append(step_desc)
        
        # Add convergence assessment
        final_confidence = thought_steps[-1].confidence_score
        if final_confidence > 0.8:
            summary_parts.append("Assessment: High confidence in final decision")
        elif final_confidence > 0.5:
            summary_parts.append("Assessment: Moderate confidence in final decision")
        else:
            summary_parts.append("Assessment: Low confidence, may need more exploration")
        
        return "\n".join(summary_parts)
    
    def _compute_convergence_metrics(self, thought_steps: List[ThoughtStep]) -> Dict[str, float]:
        """Compute convergence-related metrics"""
        if not thought_steps:
            return {}
        
        metrics = {}
        
        # Confidence stability
        confidences = [step.confidence_score for step in thought_steps]
        if len(confidences) > 1:
            confidence_variance = np.var(confidences)
            metrics['confidence_stability'] = 1.0 / (1.0 + confidence_variance)
        
        # Energy convergence (if available)
        energies = []
        for step in thought_steps:
            if 'current_energy' in step.energy_landscape:
                energies.append(step.energy_landscape['current_energy'])
        
        if len(energies) > 1:
            energy_trend = np.polyfit(range(len(energies)), energies, 1)[0]
            metrics['energy_trend'] = energy_trend
            metrics['energy_stability'] = 1.0 / (1.0 + np.var(energies))
        
        # Correction ratio effectiveness
        correction_ratios = [step.correction_ratio for step in thought_steps if step.correction_ratio is not None]
        if correction_ratios:
            metrics['avg_correction_ratio'] = np.mean(correction_ratios)
            metrics['correction_stability'] = 1.0 / (1.0 + np.var(correction_ratios))
        
        return metrics
    
    def _extract_solver_diagnostics(self) -> List[Dict[str, Any]]:
        """Extract solver diagnostics from hooks"""
        diagnostics = []
        
        for hook in self.hooks.values():
            if hasattr(hook, 'solver_diagnostics'):
                diagnostics.extend(getattr(hook, 'solver_diagnostics', []))
        
        return diagnostics
    
    def visualize_reasoning_chain(self, reasoning_chain: ReasoningChain, 
                                save_path: Optional[str] = None) -> None:
        """Visualize the reasoning chain"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MCMC Reasoning Chain Analysis', fontsize=16)
        
        # 1. Confidence trajectory
        axes[0, 0].plot(reasoning_chain.confidence_trajectory, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].scatter(reasoning_chain.decision_points, 
                          [reasoning_chain.confidence_trajectory[i] for i in reasoning_chain.decision_points],
                          color='red', s=100, zorder=5, label='Decision Points')
        axes[0, 0].set_title('Confidence Trajectory')
        axes[0, 0].set_xlabel('Reasoning Step')
        axes[0, 0].set_ylabel('Confidence Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Energy landscape
        energies = []
        for step in reasoning_chain.thought_steps:
            if 'current_energy' in step.energy_landscape:
                energies.append(step.energy_landscape['current_energy'])
        
        if energies:
            axes[0, 1].plot(energies, 'g-s', linewidth=2, markersize=6)
            axes[0, 1].set_title('Energy Landscape')
            axes[0, 1].set_xlabel('Reasoning Step')
            axes[0, 1].set_ylabel('Energy Value')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No energy data available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Energy Landscape (No Data)')
        
        # 3. Reasoning vector evolution
        reasoning_vectors = torch.stack([step.reasoning_vector for step in reasoning_chain.thought_steps])
        im = axes[1, 0].imshow(reasoning_vectors.T.numpy(), aspect='auto', cmap='viridis')
        axes[1, 0].set_title('Reasoning Vector Evolution')
        axes[1, 0].set_xlabel('Reasoning Step')
        axes[1, 0].set_ylabel('Feature Dimension')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Convergence metrics
        metrics = reasoning_chain.convergence_metrics
        if metrics:
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = axes[1, 1].bar(range(len(metric_names)), metric_values, color='skyblue')
            axes[1, 1].set_title('Convergence Metrics')
            axes[1, 1].set_xticks(range(len(metric_names)))
            axes[1, 1].set_xticklabels(metric_names, rotation=45, ha='right')
            axes[1, 1].set_ylabel('Metric Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        else:
            axes[1, 1].text(0.5, 0.5, 'No convergence metrics available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Convergence Metrics (No Data)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def export_reasoning_chain(self, reasoning_chain: ReasoningChain, 
                             export_path: str) -> None:
        """Export reasoning chain to JSON for further analysis"""
        export_data = {
            'input_shape': list(reasoning_chain.input_data.shape),
            'num_thought_steps': len(reasoning_chain.thought_steps),
            'confidence_trajectory': reasoning_chain.confidence_trajectory,
            'decision_points': reasoning_chain.decision_points,
            'reasoning_summary': reasoning_chain.reasoning_summary,
            'convergence_metrics': reasoning_chain.convergence_metrics,
            'thought_steps': []
        }
        
        for step in reasoning_chain.thought_steps:
            step_data = {
                'step_id': step.step_id,
                'layer_name': step.layer_name,
                'confidence_score': step.confidence_score,
                'energy_landscape': step.energy_landscape,
                'correction_ratio': step.correction_ratio,
                'reasoning_vector': step.reasoning_vector.tolist(),
                'metadata': step.metadata
            }
            export_data['thought_steps'].append(step_data)
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Reasoning chain exported to {export_path}")
    
    def compare_reasoning_chains(self, chains: List[ReasoningChain], 
                               labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare multiple reasoning chains"""
        if not chains:
            return {}
        
        if labels is None:
            labels = [f"Chain {i+1}" for i in range(len(chains))]
        
        comparison = {
            'num_chains': len(chains),
            'labels': labels,
            'confidence_comparison': {},
            'convergence_comparison': {},
            'decision_point_comparison': {}
        }
        
        # Compare confidence trajectories
        for i, (chain, label) in enumerate(zip(chains, labels)):
            comparison['confidence_comparison'][label] = {
                'final_confidence': chain.confidence_trajectory[-1] if chain.confidence_trajectory else 0.0,
                'avg_confidence': np.mean(chain.confidence_trajectory) if chain.confidence_trajectory else 0.0,
                'confidence_stability': np.std(chain.confidence_trajectory) if chain.confidence_trajectory else 0.0
            }
        
        # Compare convergence metrics
        for i, (chain, label) in enumerate(zip(chains, labels)):
            comparison['convergence_comparison'][label] = chain.convergence_metrics
        
        # Compare decision points
        for i, (chain, label) in enumerate(zip(chains, labels)):
            comparison['decision_point_comparison'][label] = {
                'num_decision_points': len(chain.decision_points),
                'decision_points': chain.decision_points
            }
        
        return comparison


def create_interpretability_demo():
    """Create a demonstration of the interpretability solver"""
    print("MCMC Interpretability Solver Demo")
    print("=" * 50)
    
    # This would be used with an actual enhanced MCMC model
    # For demo purposes, we'll show the structure
    
    demo_code = '''
    # Example usage with Enhanced CTM model:
    
    from models.enhanced_mcmc_layers import EnhancedCTMFenchelYoungIntegration
    from models.fenchel_young_mcmc import BinaryHypercube, MCMCConfig
    from models.mcmc_interpretability_solver import BlackBoxSolver
    
    # Create enhanced CTM model
    output_space = BinaryHypercube(dimension=4)
    config = MCMCConfig(num_chains=3, chain_length=100)
    model = EnhancedCTMFenchelYoungIntegration(
        input_dim=8,
        output_space=output_space,
        mcmc_config=config,
        use_large_neighborhood_search=True
    )
    
    # Create interpretability solver
    solver = BlackBoxSolver(model)
    
    # Extract reasoning chain
    x = torch.randn(8)
    target = torch.randint(0, 2, (4,)).float()
    reasoning_chain = solver.extract_reasoning_chain(x, target)
    
    # Analyze the reasoning
    print("Reasoning Summary:")
    print(reasoning_chain.reasoning_summary)
    
    print("\\nConvergence Metrics:")
    for metric, value in reasoning_chain.convergence_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Visualize the reasoning process
    solver.visualize_reasoning_chain(reasoning_chain, save_path="reasoning_analysis.png")
    
    # Export for further analysis
    solver.export_reasoning_chain(reasoning_chain, "reasoning_chain.json")
    '''
    
    print(demo_code)
    
    return demo_code

if __name__ == "__main__":
    create_interpretability_demo()