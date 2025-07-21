# Continuous Thought Machines (CTM) Models

## Overview

The Continuous Thought Machines (CTM) is a biologically-inspired AI model framework designed for advanced reasoning tasks, such as those in ARC-AGI-2. It integrates diffusion models with hierarchical recurrent modules, neuromodulation, long-term memory, and efficient state-space modeling. The system draws from neuroscience principles to enable adaptive, continual learning and multi-modal processing (e.g., text, images, audio).

Key features include:
- Hierarchical reasoning with high-level (slow) and low-level (fast) cycles.
- Neuromodulation for dynamic adaptation (e.g., dopamine for reward, serotonin for stability).
- Long-term memory with surprise-based storage and replay.
- Efficient neuron selection using biological strategies.
- Real-time capabilities, such as voice streaming.
- Modular utilities for positional encoding, entropy computation, and backbones for various tasks.

## File Structure and Components

Here's a breakdown of each file, its purpose, and key classes/functions:

### ctm_components.py
- **Purpose**: Contains shared core components for CTM models, including configuration, neuromodulator management, and hierarchical reasoning modules. It defines the building blocks for the overall architecture, handling continual learning, diffusion integration, and bio-inspired features.
- **Key Components**:
  - `EnhancedCTMConfig`: Comprehensive config for model parameters (e.g., dimensions, neuron selection, diffusion steps).
  - `NeuromodulatorManager`: Manages multiple neuromodulators (e.g., dopamine, serotonin) and fuses their outputs.
  - `HRM_H_Module` & `HRM_L_Module`: High-level (slow) and low-level (fast) modules for hierarchical reasoning, using Mamba blocks and attention.
  - `WorkingMemoryBuffer`: Short-term memory for recent items.
  - Utility classes like `BinarySparseAttention` and `WINASparsifier` for efficient attention.

### ctm_Diffusion_NEWNEW.py
- **Purpose**: Implements the main `EnhancedCTMDiffusion` model, combining CTM with diffusion processes for generation and denoising. It handles ultra-fast generation flows and integrates with other components for tasks like text/audio synthesis.
- **Key Components**:
  - `EnhancedCTMDiffusion`: Core model class for diffusion-based generation, using CTM for conditioning.
  - Methods for simultaneous text/audio generation and real-time processing.

### mamba_block.py
- **Purpose**: Provides an implementation of Mamba-2 blocks for state-space duality (SSD), used in hierarchical modules and efficient sequence processing.
- **Key Components**:
  - `Mamba2Block`: Self-contained block with projections, parameters, and SSD computation.
  - Functions like `ssd` and `recurrent_ssd` for efficient forward passes.

### long_term_memory.py
- **Purpose**: Implements a MemOS-inspired long-term memory system with biological features like surprise weighting, lifecycle management, and graph-based organization using Polars for efficient data handling.
- **Key Components**:
  - `MemCube`: Data structure for memory units with metadata (e.g., surprise, provenance).
  - `MemOSInspiredMemory`: Main memory class with methods for adding, retrieving, replaying, and merging memories.
  - Utilities like `MemScheduler`, `MemLifecycle`, and `MemGovernance` for dynamic management.

### neuromodulators.py
- **Purpose**: Defines classes simulating brain neuromodulators to adapt model behavior (e.g., reward boosting, stress response).
- **Key Components**:
  - Base `BaseNeuromodulator` and specific modulators like `DopamineModulator`, `SerotoninModulator`, etc.
  - Each modulates input states based on signals like reward error or uncertainty.

### realtime_voice_module.py
- **Purpose**: Integrates real-time audio streaming with the CTM model, using dynamic entropy patching for efficient voice data processing.
- **Key Components**:
  - `RealtimeVoiceStreamer`: Handles audio input/output streams and processes chunks with the model.

### modules.py
- **Purpose**: Collection of utility modules, including identity ops, UNET-style synapses, super-linear layers for neuron-level models, and backbones for tasks like MNIST or MiniGrid.
- **Key Components**:
  - `SynapseUNET` & `SuperLinear`: Core for CTM's synaptic and neuron-level processing.
  - Positional encodings like `LearnableFourierPositionalEncoding`.
  - Backbones such as `MNISTBackbone`, `MiniGridBackbone`.

### utils.py
- **Purpose**: Helper functions for computations like decay, coordinate addition, entropy calculation, and task analysis.
- **Key Components**:
  - `compute_decay`, `add_coord_dim`, `compute_normalized_entropy`.
  - `TaskAnalyzer`: Detects data modality (e.g., audio, image) for adaptive processing.

### constants.py
- **Purpose**: Defines valid constants and options for configurations (e.g., neuron selection types, attention types).
- **Key Components**:
  - Lists like `VALID_NEURON_SELECT_TYPES`, `VALID_POSITIONAL_EMBEDDING_TYPES`.

### biological_neuron_selection.py
- **Purpose**: Implements bio-inspired strategies for neuron selection, enhancing efficiency and adaptability.
- **Key Components**:
  - `BiologicalNeuronSelector`: Supports methods like Hebbian, plasticity, competitive, etc.
  - Factory `create_biological_selector` for easy integration.

## How the Files Work Together

1. **Configuration and Setup**: Start with `EnhancedCTMConfig` in ctm_components.py to set parameters (e.g., neuron selection from constants.py, positional encodings from modules.py).

2. **Core Model Flow**:
   - The main model (`EnhancedCTMDiffusion` in ctm_Diffusion_NEWNEW.py) uses components from ctm_components.py (e.g., hierarchical modules with Mamba from mamba_block.py).
   - Neuron selection uses strategies from biological_neuron_selection.py.
   - Neuromodulation from neuromodulators.py adapts states dynamically.

3. **Memory and Adaptation**:
   - Long-term memory (long_term_memory.py) stores and replays experiences, integrated into the model's forward pass.
   - Utilities (utils.py) handle entropy and task analysis to guide processing.

4. **Input/Output Handling**:
   - Backbones in modules.py process inputs (e.g., images, sequences).
   - Real-time features (realtime_voice_module.py) enable live audio integration.

5. **Training and Inference**:
   - During training, neuromodulators and biological selection adapt the model.
   - Generation uses diffusion with CTM conditioning for tasks like ARC puzzles.

This modular design allows easy extension (e.g., new modulators or selection strategies). For usage examples, see notebooks like Arc_AGI_2_Final.ipynb.
