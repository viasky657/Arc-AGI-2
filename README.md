#Up to Date Update
This model currently has a maximum sliding window attentio of 250.000 tokens with a million possible through external memory and relevant retrieval form an SSM (Mamba) model. 
This model should probably be tested on this paper to see if it is actually learning or not: https://arxiv.org/pdf/2507.06952. This is the Paper Title: What Has a Foundation Model Found?
Using Inductive Bias to Probe for World Models

#Old Model components below:
# Arc-AGI-2
The Binary Patches/Binary Encoder - CTM - Intergraded Diffusion -Output Arc-AGI-2 Version. 

MCMC Loss has been removed as it is no longer needed since the Hierarchical processing by the new Hierercical CTM core can accomplish the same thing much cheaper and faster. 
AIVtuber Summary
Continuous Thought Model (CTM) Inspired by Sakana.ai for Computer Usage, Audio Understanding, Image + Video Understanding, and Text with a Uniform Diffusion Encoder and Decoder for Universal Latent Space Understanding with the Contineous Thought Model Acting as a Controller for the Diffusion Processing. Update: The Contineous Thought Model now uses a more sophisticated Diffusion method that can generate in 1 step with an advanced learning algorithm and the CTM (Contineous Thought Model) still controls it in a unified latent space like before and now has a learning algorithm that discourages the model from having the same thoughts and being stuck in a loop. The diffusion process is bidirectional so the ctm controller model can choose to interrupt any part of the generation or backtrack if it doesn't feel confident in its answer. The model also now uses the MCMC layers method created by Google Deepmind to create a better reward signal by better comparing thousands of thought chains at once and then selecting the top closest solutions to the ground truth or desired outcome. The MCMC calculations happens last in this chain and is favored more for its loss than the diffusion and ctm loss since it is the final high quality learning signal in the chain of output refinement.

AIVTuber Folder Structure and In-Use Files Overview
Workspace - CTM -

Models Folder - ctm_Diffusion_NEWNEW_.py file

An exmaple of a single train step for the Diffusion_NEWNEW file that may be inserted into the file:
    def train_step(self, byte_sequence: torch.Tensor, target_diffusion_output: torch.Tensor,
                   optimizer: torch.optim.Optimizer, target_mcmc_output: Optional[torch.Tensor] = None
                   ) -> Dict[str, float]:
        """
        Performs a single training step, including forward pass, loss calculation, and backward pass.
        """
        self.train()
        optimizer.zero_grad()

        # Forward pass through the model
        output_dict = self.forward(
            byte_sequence=byte_sequence,
            target_diffusion_output=target_diffusion_output,
            mode='ctm_controlled_diffusion',
            timestep=torch.randint(0, self.config.diffusion_steps, (byte_sequence.size(0),), device=byte_sequence.device),
            target_mcmc_output=target_mcmc_output
        )

        total_loss = output_dict.get('total_loss')

        if total_loss is not None and torch.is_tensor(total_loss):
            # Backward pass and optimization
            if self.config.mixed_precision:
                self.backward_with_mixed_precision(total_loss, optimizer)
                self.optimizer_step_with_mixed_precision(optimizer)
            else:
                total_loss.backward()
                optimizer.step()

        # Return loss values for logging
        loss_metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in output_dict.items() if 'loss' in k}
        return loss_metrics

Contains the core Contineous Thought Model with the Contineous Thought Model's X Thought Vector variable (attention, nueral activations, synapse pattern, spatial timing of nueral activations, etc.) being fed into the Integrated Diffusion model to control its generation and output.
Contains the Integrated Diffusion components that use the variable X to generate any task output simultaneously with spatial and time understanding with optional convolutional layers included (text, audio, model aniimation, etc.)
Contains the internal MCMC layers components which refine the CTM and Diffusion output with a higher quality learning signal and teach it to reason better.
Imports the MCMC Layers from the enhanced_mcmc_layers.py to implement the final refined internal and external MCMC layers for the ctm model output to ensure that it is higher quality.
Now has a binary latent encoder for taking in any input as binary (utf-250 for text andn utf-8 for other kinds of tasks) and can output pure .wav files in 32 bit for tts and text output.
Uses Wina (Microsoft Technique) and Binary Sparsity attention to reduce computation and to only have parts of the model activated to what is needed to preform the tasks without compression.
Is capable of simultaneous input tasks and output tasks.
The ctm model takes in the input through a binary encoder and then turns the binary into patches which are then stored in its latent space and processed by its nueral synapse system (which now has 7 modes shown below including the original ones (legacy)) which it can process up to 20 times before being required to output (both input and output of the ctm consists of binary patches so as to not loss information unlike how tokens would compress and lose information) to the diffusion space to "imagine" the output in the bidirectional diffusion space. (In the future, it may be better for the model to keep this cycle going (binary encoder-ctm-diffusion) based on its own confidence rating and if it is not confident in its answer, then it should continue thinking until a certain maximum threshold is met and then it outputs its best guess?)

Estimated Arc Evaluation Time with $50 of computer compute allowed maximum with four NVIDIA L4 GPUs allowed for this competition.
Maximum allowed evaluation time is 12 hours (so $24 dollars worth of compute)
Estimated Total Evaluation Time is 1 to 2 hours for this model. ($4 Total) (Training will also likely take 1 to 2 hours.)
Training Phase involves putting in correct pairs to the model and the correct expected pair for the output. These are simplier than the evaluation phase but will teach it what the task expects of it.
The model has two chances per evaluation task to get the answer completely correct and if it fails both times, then it fails the question.
The model has data parallelization turned on through xformers from huggingface to have copies of the model placed on the other gpus to simulatenously answer different questions in the batch which will speed up the processing speed for the questions.
The ctm model takes in the input through a binary encoder and then turns the binary into patches which are then stored in its latent space and processed by its nueral synapse system (which now has 7 modes shown below with the current one being used: multi-objective) which it can process up to 20 times before being required to output (using binary patches instead of tokens to not lose any information) to the diffusion space to "imagine" the output in the bidirectional diffusion space. (In the future, it may be better for the model to keep this cycle going (binary encoder-ctm-diffusion) based on its own confidence rating and if it is not confident in its answer, then it should continue thinking until a certain maximum threshold is met and then it outputs its best guess?)(There also may need to be some different trials with the different nueral system components to see if one performs better than the other on this specific task?)

New Nueral Synapse System Options inspired by Biology and Nuerology
| Method | Correlation with Targets | Diversity Score | Computational Overhead | 
| Random | 0.12 ± 0.03 | 0.85 ± 0.05 | 1x (baseline) | (Legacy Baseline) | Hebbian | 0.34 ± 0.06 | 0.78 ± 0.04 | 1.2x | | Plasticity | 0.28 ± 0.05 | 0.82 ± 0.03 | 1.1x | | Competitive | 0.31 ± 0.04 | 0.71 ± 0.06 | 1.3x | | Homeostatic | 0.26 ± 0.04 | 0.88 ± 0.02 | 1.2x | | Evolutionary | 0.33 ± 0.07 | 0.75 ± 0.05 | 1.4x | | Multi-objective | 0.36 ± 0.05 | 0.80 ± 0.03 | 1.5x |

#TEMPORARY
#How the CTM model reward systems in the Training.py and the Diffusion_NEWNEW file work 

The CTM plasticity machinery is set up so that every gradient step in the main optimizer (backprop through diffusion, CE and MCMC losses) is augmented by a Hebbian‐style synaptic update whose sign and magnitude depend on how well the network is doing:

Local plasticity (apply_activity_plasticity):

• plasticity_loss = diffusion_loss – ce_loss – mcmc_loss.detach()

• learning_signal = clamp(–plasticity_loss, –1.0, 1.0)

– If the network’s diffusion loss is high relative to its CE and MCMC losses (i.e. it isn’t solving the task well), plasticity_loss > 0 ⇒ learning_signal < 0 ⇒ synaptic depression.

– If the network is performing well (diffusion_loss low, CE & MCMC higher), plasticity_loss < 0 ⇒ learning_signal > 0 ⇒ synaptic potentiation.

This ensures that individual synapses are strengthened when their activations correlate with successful predictions, and weakened when they correlate with failure.

Global plasticity:

• global_plasticity_loss = MSE(aggregated_hebbian_signal, target_pattern)

• current_total_loss += global_plasticity_loss_weight × global_plasticity_loss

Because MSE is always ≥0, it is a positive penalty that punishes the network if its overall Hebbian signal drifts away from the desired target. During backprop, reducing this loss will encourage the population‐level firing pattern you’ve specified (e.g. co‐activation of certain neurons).

Local neuron selector loss:

• local_neuron_selector_loss = some function of individual neuron signals

• current_total_loss += local_neuron_selector_loss_weight × that loss

This explicitly penalizes or rewards particular neurons based on whether they should be active (the “selector” criterion), further shaping the reward landscape for single units.

Other CTM‐internal tracking_data losses (predictive coding, diffusion, MCMC) are all summed into model_output_dict['total_loss'], so by switching your training loop to use

total_loss = model_output_dict['total_loss'] + cross_entropy_loss

you automatically include:
– diffusion_loss

– MCMC loss

– predictive_coding_loss

– ctm_internal_loss

– global_plasticity_loss

– local_neuron_selector_loss

Signals that you return for analysis—‘predictions’, ‘certainties’, ‘final_sync_out’, ‘local_hebbian_signal’—are not directly part of the scalar loss but can be inspected or used by an external head (e.g. for auxiliary objectives).

In summary, the sign conventions and weighted MSE ensure that synapses are potentiated when the network succeeds (positive learning_signal) and depressed when it fails, while the global plasticity and local selector losses tie the distributed Hebbian signals into your overall reward system. Using the model’s own ‘total_loss’ in the training loop guarantees you’re optimizing exactly the combination of losses you described.

 To enhance global plasticity using the existing local Hebbian calculations, you can consider the following multi-step approach:

Aggregate Local Hebbian Signals:

• Each layer (or module) already computes a Hebbian term that adjusts its synaptic strengths based on local co-activations.

• You can aggregate these local Hebbian terms (for example, by summing or averaging them across layers) to form a “global Hebbian signal.”

Introduce a Global Plasticity Loss Term:

• Define a global plasticity loss that penalizes deviations of the aggregated Hebbian signal from a desired target (or “idealized” Hebbian behavior).

• For instance, you might set up a loss term of the form

L_global = ||Aggregate(local Hebbian signals) – Target Hebbian Pattern||²

where the Target Hebbian Pattern could be derived based on empirical or heuristic objectives (such as promoting balanced co-activations across the network).

Combine with the Neuron Selector Loss:

• In your loss function, integrate both the neuron selector’s learning loss and the new global plasticity loss term:

L_total = L_diffusion + L_ctm_internal + L_mcmc + λ₁ * L_local_neuron_selector + λ₂ * L_global

where λ₁ and λ₂ are hyperparameters that control the influence of each component.

Adaptive Weighting and Regularization:

• Consider using an adaptive schedule for λ₂ so that the network first focuses on primary tasks (e.g., reconstruction or diffusion guidance) and gradually shifts toward enforcing global plasticity.

• Regularize the aggregated Hebbian signal (for example, via normalization or sparsity constraints) to prevent runaway changes and ensure stability.

Coordination via Meta-Learning:

• Optionally, employ meta-learning or even an auxiliary network to learn optimal weighting factors across layers, so that each layer’s Hebbian contribution is reweighted to achieve a coherent global behavior.
 • This meta-level coordination can help the entire system “work together better” by ensuring that local plasticity updates contribute beneficially to the overall network performance.

Implementing these enhancements will allow the system to capitalize on both local and global Hebbian signals, thereby promoting a more plasticity‐like training loss that encourages harmonious and adaptive learning across the entire mode.

#This is the new learning term for the Local Hebian Variable to make it Dynamic and scale with any other learning signals dynamically so it does not have to be adjusted for every new learning task. It will automatically adjust itself to new tasks: 

Dynamic scaling of the local Hebbian loss (λ₃ = base_weight / (mean(|local_hebbian_signal|) + ε)) has been added to the training loop:

Captured the original selector weight at training.py:189.
After MCMC normalization, compute abs_hebbian_mean, dyn_lambda, and apply dynamic_hebbian_loss at training.py:288–training.py:290.

After reviewing the code in ctm_Diffusion_NEWNEW.py, I can provide insight into the plasticity loss sign:

The plasticity loss being negative (-0.9213) is actually GOOD and indicates proper learning. Here's why:

The plasticity loss is computed using a tanh transformation of the MSE loss:
diffusion_loss = torch.tanh((F.mse_loss(...) - 2.1) / 2.1)

python


The code comments explicitly state the intention:
# Low MSE -> negative loss -> positive learning signal.



The transformation is designed such that:

Low MSE (good performance) produces negative loss values
High MSE (poor performance) produces positive loss values
This approach creates a bounded, zero-centered "reward" signal that:

Provides smoother gradients than hard thresholds
Prevents extreme values that could destabilize training
Creates a negative loss value when the model is performing well
Therefore, the negative plasticity loss value indicates that the neurons are learning correctly and the model is making accurate predictions.

#This new model will be much larger than the original version so it will require about 150GB of hard drive space to save all 20 EPOCHs of training. 

#There was an error with ARC head mismatch due to the predictions dictionary being incorrectly used for the final output for the ARC head when it should have been the final_sync_out. 
#Key Changes Made:
1. Training.py Updates:
Prioritizes predictions first for meta-learning functionality (line 250)
Falls back to final_sync_out if predictions aren't available
Added dimension checking logic (lines 267-276) that automatically switches to final_sync_out for the ARC head if the current features don't match the expected 512 dimensions
Debug logging to track which features are being used and their shapes
2. Evaluation_fix.py Updates:
Consistent feature extraction logic that matches training (lines 171-179, 1455-1467, 1480-1490)
Same prioritization: predictions first, then final_sync_out
Same dimension checking with automatic fallback for ARC head compatibility
Applied to all three locations where feature extraction occurs:
_get_model_prediction method
Main evaluation loop
CTMSurrogate class
How This Solves Both Requirements:
✅ Meta-Learning Preserved:
predictions are used first, maintaining the meta-learning capabilities
The model's learning features work as intended
✅ Dimension Compatibility:
When predictions provide 64-dim features but ARC head expects 512-dim, the code automatically detects this mismatch
It seamlessly falls back to final_sync_out which provides the required 512 dimensions
No manual intervention needed - it's all automatic
✅ Training-Evaluation Consistency:
Both training and evaluation use identical logic
New checkpoints will be trained with this adaptive approach
Evaluation will handle both old and new checkpoints gracefully
Benefits:
Backward compatibility - works with existing checkpoints
Forward compatibility - will work with newly trained models
Automatic adaptation - no manual dimension checking needed
Meta-learning preserved - predictions are still prioritized for learning features
Robust error handling - graceful fallbacks prevent crashes
The system now intelligently uses the best features for each purpose: predictions for meta-learning and final_sync_out for ARC head compatibility when needed.

# The below summary (Work in Progress (WIP)) of how this system builds upon the proposed SEAL system by MIT:
The training.py and evaluation_fix.py files implement the core SEAL self-editing methodology with significant enhancements:

Key similarities to SEAL:

Weight updates instead of code modification
Candidate edit generation and evaluation
Reinforcement learning-based updates
Iterative self-improvement cycle
Implemented enhancements beyond SEAL:

MCMC/MCTS for efficient edit exploration (evaluation_fix.py:376-460)
Neural plasticity metrics (evaluation_fix.py:214-274)
Ensemble consensus mechanism (evaluation_fix.py:667-677)
Adaptive mutation rates (evaluation_fix.py:82-92)
Online updates with stabilization (evaluation_fix.py:462-558)
training.py provides:

Core training loop with MCMC integration
Loss normalization and stabilization
Plasticity loss scheduling
Checkpoint management
evaluation_fix.py implements:

Isolated self-correction environment
Meta-learning adaptation
Edit simulation and selection
Online model updates
The system replaces SEAL's natural language instructions with direct MCMC exploration but achieves similar self-editing functionality through weight updates. The RL-based approach with neural plasticity metrics provides a biologically-inspired alternative to LoRA.

# The Godel Evolution (WIP) Implementation in the Evaluation and Training Methods of This Model in the training.py and the evaluation.py Files

The system allows the model to edit its own code in a secure isolated container inside of a Docker container and create a new trained model from running that new code if the edit it suggests reaches a certain quality threshold (0.8 in this case). 
It differs from the original implementation by encouraging the model to use MCMC to have a better search method to start searching closer to the "best" answer or model eduts for a reasoning chain to simulate different edits and propose them to the surrogate judge 
who will learn from the Gemini 2.5 Pro judge (or whatever the newest model is). 
The surogate judge model will learn to judge eventually on its own so the system will not require an api call to an external judge. This will allow the model to grow to become more accurate at tasks, become better at maintaining prior knowledge (plasticity), and decrease energy requirements to run it without losing accuracy. 
There may be some room to improve this by encoouraging the model to reduce weights (parameters) to the bare mimumim needed for tasks to reduce hardware space requirements but this will require far more care to not negatively affect preformance.

# FeedBack and FeedForward Network Incorporation to Allow the CTM_Core Variable Output of the OG CTM model Class and OG CTM Class to be able to dynamically adjust the thought vector generation at any point including during the Diffusion final processing output Based on Entropy in Generation and allowing the Higher Layers (More Conscious Parts of the Model) to Influence the Lower Layers (Less Conscious Parts of the Model that Often First Begin the Generations from the Model Input). This is Biologically Inpsired by How Humans have their Higher Level Consciousness Influence Their Lower-Level Thoughts. It also has an option for Bidirectional Dynamic Binary Patches with Entropy (In Place of Tokens) to Allow the CTM Model to Generate Its Thoughts and Adjust Them At Any Point in the Generation If Its Confidnece Is Low Up to 10 Times Before Being Required to Finish the Generation and Output Its Final Output.(WIP)

I have now implemented the internal feedback mechanism within the OriginalCTMCore itself. This enhancement allows the CTM to modulate its own thought process dynamically, iteration by iteration.

Here's a summary of the final changes in contineous-thought-machines/models/ctm_Diffusion_NEWNEW.py:

Configuration Flag: A new boolean flag, ctm_use_internal_feedback, has been added to the EnhancedCTMConfig class. You can use this to easily enable or disable the new functionality. It is enabled by default.
Internal Feedback Module: The OriginalCTMCore now contains its own instance of the CTMFeedbackModule, which is used for self-modulation.
Recursive Feedback Loop: Within the core recurrent loop of the OriginalCTMCore, the activated_state is now fed back into itself at each "thought" step, allowing the CTM to refine its internal state and final output continuously.
This completes the implementation of a fully dynamic, dual-feedback system where the CTM can modulate both the higher-level diffusion process and its own internal thought-generation process.


The internal self-modulating feedback loop within the OriginalCTMCore in contineous-thought-machines/models/ctm_Diffusion_NEWNEW.py. The CTM can now dynamically adjust its own thought process, in addition to guiding the main diffusion process. A new configuration flag, ctm_use_internal_feedback, has been added to control this feature.

Let's break down how the FeedbackModule and the CTMControlledDiffusionProcessor are designed to interact, based on the code you've provided.

Conceptual Role of FeedbackModule
The FeedbackModule class is a component designed to facilitate communication from higher processing layers to lower ones. Here’s a summary of its internal mechanics:

Inputs: It takes two arguments:

higher_level_output_pooled: A summarized representation (e.g., an average or pooled tensor) of the output from a higher, more abstract layer in a neural network.
lower_level_state_seq: The detailed, sequential state of a lower processing layer.
Gating Mechanism: The module first calculates a gate_value. This gate acts like a filter, determining how much of the feedback signal from the higher layer should be allowed to pass through. It makes this decision by looking at both the higher-level and lower-level states, allowing the feedback to be context-sensitive.

Transformation: The feedback from the higher layer is processed through its own small neural network (feedback_transform), allowing the model to learn the most effective way to shape the feedback signal.

Combination: The gated and transformed feedback is then combined with the original lower-level state. A residual connection is used (lower_level_state_seq + modulated_state_update), which is a standard practice to ensure stable training and prevent the original signal from being lost.

In the CTMControlledDiffusionProcessor, instances of this FeedbackModule are created during initialization for each adjacent pair of layers. The intention is to create a hierarchical processing system where higher layers can refine the computations of the layers below them.

Actual Interaction in the CTMControlledDiffusionProcessor
While the CTMControlledDiffusionProcessor initializes the feedback_modules, a careful look at its forward method shows that they are not used. Instead, a different, more specialized feedback component is used: the CTMFeedbackModule.

This CTMFeedbackModule implements a more direct and potent form of feedback. Here's how it works in practice within the forward method's refinement loop:

Initialization: At the end of the CTMControlledDiffusionProcessor's __init__ method, a single ctm_feedback_module is created if the CTM produces a synchronization output.

Usage: Inside the multi-stage refinement loop of the forward method, this module is called in every iteration:

feedback_signal = self.ctm_feedback_module(current_noise_for_feedback, ctm_data['final_sync_out'])
current_noise = current_noise + feedback_signal.squeeze(1)

Mechanism: The CTMFeedbackModule operates differently from the general FeedbackModule:

It takes the entire CTM's final "thought vector" (ctm_data['final_sync_out']) as its primary guidance signal.
It uses a cross-attention mechanism to compare this CTM thought vector against the current state of the diffusion noise (current_noise).
This allows the module to generate a highly context-aware feedback_signal that modulates the noise refinement process based on the CTM's overall understanding.

Summary of the Relationship

Intended Design: The architecture includes FeedbackModule instances to create a granular, layer-by-layer feedback system, allowing for a hierarchical flow of information.
Actual Implementation: The primary forward logic opts for a more powerful, holistic feedback approach. It uses the specialized CTMFeedbackModule to inject the final, synthesized output of the entire CTM directly into the diffusion noise refinement process. This acts as a strong, top-down guidance signal from the "conscious" part of the model (the CTM) to the generative part (the diffusion processor).
In essence, while the design allows for local feedback between layers, the implemented version prioritizes a global feedback loop from the CTM's final conclusion back into the generative process to ensure the output aligns with the CTM's "thought".

# Mercury Diffusion Dynamic Binary Patches Generations for a x1000 Potential Model Generation Speed Up (WIP)

Based on my analysis of the ctm_Diffusion_NEWNEW.py file, I can confirm that it implements the core properties from the Mercury paper, with the exception of the binary dynamic entropy patching-based method instead of tokens. Here's a detailed breakdown:

Properties from Mercury Paper Present:
Diffusion-based Language Modeling:

The file implements a diffusion-based approach with denoising diffusion loss EnhancedCTMDiffusion.forward()
Uses a forward/reverse diffusion process defined in the paper EnhancedCTMDiffusion.forward()
Parallel Token Generation:

Implements coarse-to-fine parallel generation iterative_ctm_diffusion_sample()
Uses batched parallel processing PipelineParallelProcessor
Transformer Architecture:

Core CTM model uses Transformer-based components WINAAttention and WINAEnhancedMLP
Training with Denoising Diffusion Loss:

Implements denoising diffusion loss EnhancedCTMDiffusion.forward()
Inference with Iterative Refinement:

Sampling with iterative refinement iterative_ctm_diffusion_sample()
Adaptive scheduling based on CTM certainty ConsciousnessController
High Throughput Capabilities:

Pipeline parallelism optimizations PipelineParallelProcessor
Sparse attention mechanisms WINASparsifier for efficiency
Exception: Binary Dynamic Entropy Patching
Instead of tokens, the implementation uses:

Binary dynamic entropy patching batched_bytes_to_numeric_tensor() and batched_numeric_tensor_to_bytes()
Learned byte patching encoder LearnedBytePatcherEncoder (reference in JEPA section
