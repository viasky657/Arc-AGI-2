import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from torch.quantization import FakeQuantize

def quantize_adaptive(tensor, bits):
    """Dynamically quantizes a tensor to a given bitwidth."""
    if bits <= 0:
        raise ValueError("bits must be positive")
    q_min = -(1 << (bits - 1))
    q_max = (1 << (bits - 1)) - 1
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (q_max - q_min)
    if scale.abs() < 1e-6:
        scale = 1e-6
    zero_point = q_min - (min_val / scale)
    zero_point = torch.clamp(zero_point.round(), q_min, q_max).int()
    q_tensor = torch.quantize_per_tensor(tensor, scale, zero_point, torch.qint8)
    return q_tensor, scale, zero_point

def dequantize_adaptive(q_tensor, scale, zero_point):
    """Dequantizes a tensor using its scale and zero-point."""
    return q_tensor.dequantize()

class BitwidthAdapter(nn.Module):
    def __init__(self, d_model, min_bits=4, max_bits=8):
        super().__init__()
        self.bitwidth_predictor = nn.Linear(d_model, 1)
        self.min_bits = min_bits
        self.max_bits = max_bits

    def forward(self, task_embedding):
        bitwidth_logit = self.bitwidth_predictor(task_embedding)
        bitwidth = self.min_bits + (self.max_bits - self.min_bits) * torch.sigmoid(bitwidth_logit)
        return bitwidth.round().int()

class QuantizationPolicyNetwork(nn.Module):
    def __init__(self, input_dim, num_components=10):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_components * 3)  # bitwidth, scale, zero_point per component
        )
        self.num_components = num_components

    def forward(self, task_embedding):
        params = self.policy_net(task_embedding)
        return params.view(-1, self.num_components, 3)  # (batch, components, [bitwidth, scale, zero_point])

def load_quantized_state(self, quantized_state_dict):
    """
    Load pre-quantized state dict while preparing for meta-learning.
    """
    # Load the quantized weights
    self.load_state_dict(quantized_state_dict, strict=False)
    
    # Prepare for dequantization during adaptation
    self.quantized = True
    print("Loaded quantized model. Use dequantize_for_adaptation() for meta-learning tasks.")

def dequantize_for_adaptation(self):
    """
    Temporarily dequantize the model for meta-learning adaptation.
    """
    if hasattr(self, 'quantized') and self.quantized:
        # Convert parameters to float32 for full precision adaptation
        for param in self.parameters():
            param.data = param.data.float()
        self.quantized = False
        print("Model dequantized for adaptation.")
    else:
        print("Model is already in full precision.")

def quantize_after_adaptation(self, task_embedding=None):
    """
    Re-quantize the model after adaptation, optionally using adaptive bitwidth and policy.
    """
    if not hasattr(self, 'quantized') or not self.quantized:
        if task_embedding is not None and hasattr(self, 'bitwidth_adapter'):
            # Use BitwidthAdapter for dynamic bitwidth
            bitwidth = self.bitwidth_adapter(task_embedding)
            
            # Use QuantizationPolicyNetwork for per-component params
            if hasattr(self, 'quant_policy_net') and self.quant_policy_net:
                quant_params = self.quant_policy_net(task_embedding)
                # Note: This is a placeholder for a more sophisticated policy application
                scale = quant_params[:, :, 1].mean()
                zero_point = quant_params[:, :, 2].mean().round().int()
            else:
                scale, zero_point = 1.0, 0
            
            print(f"Applying adaptive quantization with bitwidth {bitwidth.item()} and custom params")
            quantized_model = self
            for name, mod in quantized_model.named_modules():
                if isinstance(mod, nn.Linear):
                    q_weight, _, _ = quantize_adaptive(mod.weight.data, bitwidth)
                    mod.weight.data = dequantize_adaptive(q_weight, scale, zero_point)
        else:
            # Default post-training quantization
            quantized_model = torch.quantization.quantize_dynamic(
                self, {nn.Linear}, dtype=torch.qint8
            )
        
        self.load_state_dict(quantized_model.state_dict())
        self.quantized = True
        print("Model re-quantized after adaptation.")
    else:
        print("Model is already quantized.")

class SelectiveQuantizer(nn.Module):
    """Enhanced selective quantizer with variable bitwidth and adaptive threshold."""
    def __init__(self, min_bits=2, max_bits=8, num_bins=3):
        super().__init__()
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.num_bins = num_bins  # Number of quantization levels

    def forward(self, weight, scores):
        """
        Apply enhanced selective quantization with variable bitwidth using quantize_adaptive.
        
        Args:
            weight (torch.Tensor): Weight tensor [out_features, in_features]
            scores (torch.Tensor): Importance scores [in_features]
            
        Returns:
            torch.Tensor: Selectively quantized weights
        """
        # Adaptive threshold: use median for robustness
        adaptive_threshold = torch.median(scores)
        
        # Sort scores and create bins
        sorted_scores, indices = torch.sort(scores)
        bin_size = len(sorted_scores) // self.num_bins
        bin_thresholds = [sorted_scores[i * bin_size] for i in range(1, self.num_bins)]
        
        # Assign bitwidths: higher scores get more bits
        bitwidths = torch.linspace(self.min_bits, self.max_bits, self.num_bins + 1).int()
        
        # Create per-column bitwidth assignment
        column_bits = torch.zeros_like(scores, dtype=torch.int)
        for i in range(self.num_bins):
            if i == 0:
                mask = scores <= bin_thresholds[0]
            elif i == self.num_bins - 1:
                mask = scores > bin_thresholds[-1]
            else:
                mask = (scores > bin_thresholds[i-1]) & (scores <= bin_thresholds[i])
            column_bits[mask] = bitwidths[i]
        
        # Group-wise quantization per column using quantize_adaptive
        new_weight = weight.clone()
        for col in range(weight.shape[1]):
            col_weight = weight[:, col]  # 1D tensor
            bits = column_bits[col]
            
            if bits == self.max_bits:  # Skip quantization for most important
                continue
                
            q_col, scale, zero_point = quantize_adaptive(col_weight, bits)
            deq_col = dequantize_adaptive(q_col, scale, zero_point)
            
            new_weight[:, col] = deq_col
        
        return new_weight

def ssd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, x: torch.Tensor, chunk_size: int, initial_h: torch.Tensor = None, confidence_level: str = 'medium') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len, p = x.shape
    n = b.shape[-1]
    num_chunks = math.ceil(seq_len / chunk_size)
    y = torch.zeros(batch, seq_len, p, device=x.device, dtype=x.dtype)
    if initial_h is None:
        h = torch.zeros(batch, n, p, device=x.device, dtype=x.dtype)
        deltas = []
    else:
        h = initial_h

    @torch.compile
    def chunk_compute(ac: torch.Tensor, bc: torch.Tensor, cc: torch.Tensor, xc: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cs = ac.shape[1]
        log_ac = torch.log(ac.clamp(min=1e-10))  # For numerical stability
        cum_log = torch.cumsum(log_ac, dim=1)
        log_temp = cum_log[:, :, None] - cum_log[:, None, :]
        mask = torch.tril(torch.ones((cs, cs), device=x.device, dtype=torch.bool))[None, :, :].to(log_temp.dtype)
        L = torch.exp(log_temp) * mask
        cb = torch.bmm(cc, bc.transpose(1, 2))
        M = L * cb
        local_y = torch.bmm(M, xc)
        initial_contrib = torch.exp(cum_log[:, :, None]) * torch.einsum('bcn,bnp->bcp', cc, h)
        log_weights = cum_log[:, -1, None] - cum_log
        weights = torch.exp(log_weights)
        sum_terms = torch.sum(weights[:, :, None, None] * (bc[:, :, :, None] * xc[:, :, None, :]), dim=1)
        initial_part = torch.exp(cum_log[:, -1, None, None]) * h
        new_h = initial_part + sum_terms
        return local_y, initial_contrib, new_h

    for cidx in range(num_chunks):
        start = cidx * chunk_size
        end = min(start + chunk_size, seq_len)
        cs = end - start
        if cs == 0:
            continue
        ac = a[:, start:end].contiguous()
        bc = b[:, start:end].contiguous()
        cc = c[:, start:end].contiguous()
        xc = x[:, start:end].contiguous()
        local_y, initial_contrib, h = chunk_compute(ac, bc, cc, xc, h)
        y[:, start:end] = local_y + initial_contrib

    return y, h

def recurrent_ssd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, x: torch.Tensor, initial_h: torch.Tensor = None, confidence_level: str = 'medium') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len, p = x.shape
    n = b.shape[-1]
    y = torch.zeros(batch, seq_len, p, device=x.device, dtype=x.dtype)
    if initial_h is None:
        h = torch.zeros(batch, n, p, device=x.device, dtype=x.dtype)
    else:
        h = initial_h
        deltas = []
        prev_h = h.clone()
    for t in range(seq_len):
        h = a[:, t, None, None] * h + b[:, t, :, None] * x[:, t, None, :]
        delta = torch.norm(h - prev_h, dim=(-1, -2)).mean()
        deltas.append(delta)
        prev_h = h.clone()
        y[:, t] = torch.einsum('bn,bnp->bp', c[:, t], h)
    if deltas:
        variance = torch.var(torch.stack(deltas))
        confidence = torch.exp(-variance)
        thresholds = {'critical': 0.99, 'medium': 0.8, 'low': 0.5}
        threshold = thresholds.get(confidence_level, 0.8)
        if confidence < threshold:
            y = y * 0
    else:
        confidence = torch.tensor(1.0, device=y.device)
    return y, h, confidence

class Mamba2Block(nn.Module):
    """
    A self-contained Mamba-2 block implementation with State Space Duality (SSD).
    This incorporates the duality between SSM and attention-like computation for efficiency.
    """
    def __init__(self, d_model: int, d_state: int = 64, d_head: int = 64, expand: int = 2, chunk_size: int = 256, config=None):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.d_state = d_state
        self.d_head = d_head
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.n_heads = self.d_inner // d_head
        self.chunk_size = chunk_size

        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        param_dim = 1 + 2 * d_state
        self.param_proj = nn.Linear(d_model, self.n_heads * param_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Quantization-aware training
        self.input_fake_quant = FakeQuantize()
        self.output_fake_quant = FakeQuantize()
        self.quant_aware = False
        self.base_thresholds = {'critical': 0.99, 'medium': 0.8, 'low': 0.5}
        self.confidence_thresholds = {k: 0.0 for k in self.base_thresholds}
        self.initial_epochs = 5
        self.current_epoch = 0

        # Quantization setup
        if self.config and self.config.ctm_use_qat:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
        
        if self.config and getattr(self.config, 'ctm_adaptive_quantization', False):
            self.bitwidth_adapter = BitwidthAdapter(self.config.d_model, self.config.ctm_quant_min_bits, self.config.ctm_quant_max_bits)
        else:
            self.bitwidth_adapter = None
        
        if self.config and getattr(self.config, 'ctm_quant_policy_search', False):
            self.quant_policy_net = QuantizationPolicyNetwork(self.config.d_model)
        else:
            self.quant_policy_net = None

        if self.config and getattr(self.config, 'ctm_selective_quantization', False):
            self.selective_quantizer = SelectiveQuantizer(self.config.ctm_quant_min_bits, self.config.ctm_quant_max_bits)
        else:
            self.selective_quantizer = None


    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor = None, confidence_level: str = 'medium') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l, d = x.shape
    
        if self.quant_aware and self.training:
            x = self.input_fake_quant(x)
    
        new_b = b * self.n_heads
        if hidden_state is None:
            hidden_state = torch.zeros(b, self.n_heads, self.d_state, self.d_head, device=x.device, dtype=x.dtype)
        hidden_state_flat = hidden_state.reshape(new_b, self.d_state, self.d_head)
    
        # Selective quantization on weights
        if self.selective_quantizer and self.training:
            with torch.no_grad():
                in_scores = x.abs().mean(dim=(0,1))
                param_scores = x.abs().mean(dim=(0,1))
            in_proj_weight = self.selective_quantizer(self.in_proj.weight, in_scores)
            param_proj_weight = self.selective_quantizer(self.param_proj.weight, param_scores)
        else:
            in_proj_weight = self.in_proj.weight
            param_proj_weight = self.param_proj.weight
            out_proj_weight = self.out_proj.weight

        x_and_res = F.linear(x, in_proj_weight, self.in_proj.bias)
        input_x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        input_x = input_x.view(b, l, self.n_heads, self.d_head)
    
        params = F.linear(x, param_proj_weight, self.param_proj.bias)
        params = params.view(b, l, self.n_heads, 1 + 2 * self.d_state)
        log_dt = params[..., 0]
        B = params[..., 1 : 1 + self.d_state]
        C = params[..., 1 + self.d_state : ]
    
        dt = F.softplus(log_dt)
        a = torch.exp(-dt)  # (b, l, n_heads)
    
        # Flatten for ssd
        a_flat = a.reshape(new_b, l)
        B_flat = B.reshape(new_b, l, self.d_state)
        C_flat = C.reshape(new_b, l, self.d_state)
        x_flat = input_x.reshape(new_b, l, self.d_head)
    
        if self.training:
            y_flat, final_h_flat, confidence = ssd(a_flat, B_flat, C_flat, x_flat, self.chunk_size, initial_h=hidden_state_flat, confidence_level=confidence_level)
        else:
            y_flat, final_h_flat, confidence = recurrent_ssd(a_flat, B_flat, C_flat, x_flat, initial_h=hidden_state_flat, confidence_level=confidence_level)
    
        y = y_flat.view(b, l, self.n_heads, self.d_head)
        y = y.reshape(b, l, self.d_inner)
    
        output = y * F.silu(res)
        if self.selective_quantizer and self.training:
            with torch.no_grad():
                out_scores = output.abs().mean(dim=(0,1))
            out_proj_weight = self.selective_quantizer(self.out_proj.weight, out_scores)
        else:
            out_proj_weight = self.out_proj.weight
        output = F.linear(output, out_proj_weight, self.out_proj.bias)
    
        if self.quant_aware and self.training:
            output = self.output_fake_quant(output)
    
        final_h = final_h_flat.reshape(b, self.n_heads, self.d_state, self.d_head)
        
        quantize = (self.config.quant_enabled_training and self.training) or \
                   (self.config.quant_enabled_inference and not self.training)

        if quantize:
            if self.config.ctm_use_qat:
                x = self.quant(x)
            
            if self.config.ctm_adaptive_quantization and self.bitwidth_adapter:
                with torch.no_grad():
                    task_embedding = x.mean(dim=1)
                    bits = self.bitwidth_adapter(task_embedding)
                
                if self.config.ctm_quant_policy_search and self.quant_policy_net:
                    with torch.no_grad():
                        policy_params = self.quant_policy_net(task_embedding)
                        scale = policy_params[:, :, 1].mean()
                        zero_point = policy_params[:, :, 2].mean()
                else:
                    scale, zero_point = 1.0, 0.0  # Defaults
                
                q_x, _, _ = quantize_adaptive(x, bits)
                x = dequantize_adaptive(q_x, scale, zero_point)
            
            if self.config.ctm_use_qat:
                output = self.dequant(output)
    
        return output, final_h, confidence

    def enable_quant_aware_training(self, enable: bool = True) -> None:
        self.quant_aware = enable
