import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt

class CTMTracker:
    def __init__(self):
        self.pre_activations = []
        self.post_activations = []
        self.synch_out = []
        self.synch_action = []
        self.attention = []
        self.dopamine_errors = []
        self.plastic_adjustments = []
        self.pc_losses = []
        self.state_traces = []
        self.activated_states = []
        self.neuromodulators = []
        self.hierarchical_projections = []

    def track_step(self, state_trace, activated_state, synch_action, synch_out, attn_weights=None,
                   dopamine_error=None, plastic_adjustment=None, pc_loss=None, neuromodulators=None, hierarchical_projection=None):
        self.state_traces.append(state_trace.detach().cpu().numpy())
        self.activated_states.append(activated_state.detach().cpu().numpy())
        self.pre_activations.append(state_trace[:,:,-1].detach().cpu().numpy())
        self.post_activations.append(activated_state.detach().cpu().numpy())
        self.synch_action.append(synch_action.detach().cpu().numpy())
        self.synch_out.append(synch_out.detach().cpu().numpy())
        if attn_weights is not None:
            self.attention.append(attn_weights.detach().cpu().numpy())
        if dopamine_error is not None:
            self.dopamine_errors.append(dopamine_error.detach().cpu().numpy())
        if plastic_adjustment is not None:
            self.plastic_adjustments.append(plastic_adjustment.detach().cpu().numpy())
        if pc_loss is not None:
            self.pc_losses.append(pc_loss.detach().cpu().item())  # Assuming scalar
        if neuromodulators is not None:
            self.neuromodulators.append(neuromodulators.detach().cpu().numpy())
        if hierarchical_projection is not None:
            self.hierarchical_projections.append(hierarchical_projection.detach().cpu().numpy())


    def get_tracking_data(self):
        return {
            'pre_activations': np.array(self.pre_activations),
            'post_activations': np.array(self.post_activations),
            'synch_action': np.array(self.synch_action),
            'synch_out': np.array(self.synch_out),
            'attention': np.array(self.attention) if self.attention else None,
            'dopamine_errors': np.array(self.dopamine_errors) if self.dopamine_errors else None,
            'plastic_adjustments': np.array(self.plastic_adjustments) if self.plastic_adjustments else None,
            'pc_losses': np.array(self.pc_losses) if self.pc_losses else None,
            'state_traces': np.array(self.state_traces),
            'activated_states': np.array(self.activated_states),
            'neuromodulators': np.array(self.neuromodulators) if self.neuromodulators else None,
            'hierarchical_projections': np.array(self.hierarchical_projections) if self.hierarchical_projections else None
        }

    def visualize_pulsing(self, output_path='neuron_pulsing.gif'):
        """
        Generates a GIF visualizing the pulsing activity of neurons over time.
        """
        if not self.activated_states:
            print("No tracking data to visualize.")
            return

        frames = []
        for i, activated_state in enumerate(self.activated_states):
            fig, ax = plt.subplots()
            # Assuming activated_state is (batch, neurons), we take the first batch item
            ax.imshow(activated_state[0][np.newaxis, :], cmap='hot', aspect='auto')
            ax.set_title(f"Neuron Activations at Step {i}")
            ax.set_xlabel("Neuron Index")
            ax.set_yticks([])
            
            # Save frame to a buffer
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            plt.close(fig)

        imageio.mimsave(output_path, frames, fps=5)
        print(f"Visualization saved to {output_path}")