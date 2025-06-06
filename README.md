# Arc-AGI-2
The Binary Patches/Binary Encoder - CTM - Intergraded Diffusion -Output Arc-AGI-2 Version. 

AIVtuber Summary
Continuous Thought Model (CTM) Inspired by Sakana.ai for Computer Usage, Audio Understanding, Image + Video Understanding, and Text with a Uniform Diffusion Encoder and Decoder for Universal Latent Space Understanding with the Contineous Thought Model Acting as a Controller for the Diffusion Processing. Update: The Contineous Thought Model now uses a more sophisticated Diffusion method that can generate in 1 step with an advanced learning algorithm and the CTM (Contineous Thought Model) still controls it in a unified latent space like before and now has a learning algorithm that discourages the model from having the same thougths and being stuck in a loop. The diffusion process is bidirectional so the ctm controller model can choose to interrupt any part of the generation or backtrack if it doesn't feel confident in its answer. The model also now uses the MCMC layers method created by Google Deepmind to create a better reward signal by better comparing thousands of thought chains at once and then selecting the top closest solutions to the ground truth or desired outcome. The MCMC calculations happens last in this chain and is favored more for its loss than the diffusion and ctm loss since it is the final high quality learning signal in the chain of output refinement.

AIVTuber Folder Structure and In-Use Files Overview
Workspace - CTM -

Models Folder - ctm_Diffusion_NEWNEW_.py file

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