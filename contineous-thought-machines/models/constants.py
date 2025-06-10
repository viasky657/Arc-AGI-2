VALID_NEURON_SELECT_TYPES = [
    'first-last', 'random', 'random-pairing',  # Legacy
    # Biologically-inspired types
    'bio_hebbian', 'bio_plasticity', 'bio_competitive', 'bio_homeostatic',
    'bio_evolutionary', 'bio_stdp', 'bio_criticality', 'bio_multi_objective',
    # Hybrid approaches
    'adaptive_random', 'performance_guided', 'task_aware'
]

VALID_BACKBONE_TYPES = [
    f'resnet{depth}-{i}' for depth in [18, 34, 50, 101, 152] for i in range(1, 5)
] + ['shallow-wide', 'parity_backbone', 'uniform_diffusion_backbone']

VALID_POSITIONAL_EMBEDDING_TYPES = [
    'learnable-fourier', 'multi-learnable-fourier',
    'custom-rotational', 'custom-rotational-1d'
]

ARC_TRAIN_DIR = "./data/training"
ARC_EVAL_DIR = "./data/evaluation"
