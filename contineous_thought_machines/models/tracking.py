import numpy as np
import torch

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

    def track_step(self, state_trace, activated_state, synch_action, synch_out, attn_weights=None, 
                   dopamine_error=None, plastic_adjustment=None, pc_loss=None):
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
            'activated_states': np.array(self.activated_states)
        }