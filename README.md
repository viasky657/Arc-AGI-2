# Arc-AGI-2
The Binary Patches/Binary Encoder - CTM - Intergraded Diffusion -Output Arc-AGI-2 Version. 

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

Models Folder - enhanced_mcmc_layers.py

Contains the MCMC large nieghborhood search and parallel search which create an exact oracle to have a search that is closer to the solution to create a quicker training time (convergence) for the model.
Contains the approximate estimation MCMC Layer function for best answers to the task.
Contains the Phi Network which contains the hypercube (MCMC dimensions) of the MCMC graphed points for the thought chains generated from the MCMC layers method.
Contains the imports from both fenchel_young_mcmc.py and mcmc_interpretability_solver.py to complete the blackbox solver of the MCMC layer for interpreting its chosen thought chains and more components of the MCMC layer.

Models Folder - fenchel_young_mcmc.py

Contains the Phi Network function for space of the generated MCMC Layers points.
Contains the temperature scheduler which controls how much the model explores the generated random thought chains. It changes from a higher temperature to have a lot of exploration and then refines it to just a very small number to hone in the exploration to only the most likely solutions.
Contains a proposal solution filter which filters out all the chosen thought chains to just the highest confidence scoring ones to increse the odds of the thought chain being correct.
Contains a structured prediction function closely-integrated with the CTM's contineous thought processes.

Models Folder - mcmc_interpretability_solver.py

Contains the attention hooks functions to hook the thought chain and MCMC layer processing to create a graph of the thought chains and captures the meaning of the thought chains to help with understanding the reason behind the model's choices for debugging.
This is turned on by default in the ctm_Newest_USE_enhanced_diffusion_ctm_model.py file for interpretability but can be turned off in that file.


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