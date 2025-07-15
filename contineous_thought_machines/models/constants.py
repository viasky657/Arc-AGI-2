"""
Constants for the CTM (Continuous Thought Machine) models.
This file defines valid options for various configuration parameters.
"""

# Valid neuron selection types for the CTM core
VALID_NEURON_SELECT_TYPES = [
    # Legacy types
    'first-last', 'random', 'random-pairing',
    
    # Biologically-inspired types
    'bio_hebbian', 'bio_plasticity', 'bio_competitive', 'bio_homeostatic',
    'bio_evolutionary', 'bio_stdp', 'bio_criticality', 'bio_multi_objective',
    
    # Hybrid approaches
    'adaptive_random', 'performance_guided', 'task_aware'
]

# Valid positional embedding types
VALID_POSITIONAL_EMBEDDING_TYPES = [
    'learnable-fourier', 'multi-learnable-fourier',
    'custom-rotational'
]

# Valid attention types
VALID_ATTENTION_TYPES = [
    'standard', 'binary_sparse', 'WINA'
]

# Valid activation functions
VALID_ACTIVATIONS = [
    'relu', 'gelu', 'swish', 'mish', 'leaky_relu'
]

# Valid noise schedules for diffusion
VALID_NOISE_SCHEDULES = [
    'linear', 'cosine', 'sigmoid'
]

# Valid replay policies for memory
VALID_REPLAY_POLICIES = [
    'simple_replay', 'surprise_weighted_replay', 'usefulness_replay'
]
