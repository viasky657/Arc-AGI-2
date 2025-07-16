import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from torch.quantization import FakeQuantize

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
    def __init__(self, d_model: int, d_state: int = 64, d_head: int = 64, expand: int = 2, chunk_size: int = 256):
        super().__init__()
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
        if config.ctm_use_qat:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
        
        if config.ctm_adaptive_quantization:
            self.bitwidth_adapter = BitwidthAdapter(config.d_model, config.ctm_quant_min_bits, config.ctm_quant_max_bits)
        
        if config.ctm_quant_policy_search:
            self.quant_policy_net = QuantizationPolicyNetwork(config.d_model)
        
        if config.ctm_selective_quantization:
            self.selective_quantizer = SelectiveQuantizer(config.ctm_quant_min_bits, config.ctm_quant_max_bits)


    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor = None, confidence_level: str = 'medium') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l, d = x.shape

        if self.quant_aware and self.training:
            x = self.input_fake_quant(x)

        new_b = b * self.n_heads
        if hidden_state is None:
            hidden_state = torch.zeros(b, self.n_heads, self.d_state, self.d_head, device=x.device, dtype=x.dtype)
        hidden_state_flat = hidden_state.reshape(new_b, self.d_state, self.d_head)

        x_and_res = self.in_proj(x)
        input_x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        input_x = input_x.view(b, l, self.n_heads, self.d_head)

        params = self.param_proj(x)
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
        output = self.out_proj(output)

        if self.quant_aware and self.training:
            output = self.output_fake_quant(output)

        final_h = final_h_flat.reshape(b, self.n_heads, self.d_state, self.d_head)

        # Quantization in forward
        if self.config.ctm_use_qat and self.training:
            x = self.quant(x)
        
        if self.config.ctm_adaptive_quantization and self.bitwidth_adapter:
            task_embedding = x.mean(dim=0)
            bits = self.bitwidth_adapter(task_embedding)
            
            if self.config.ctm_quant_policy_search and self.quant_policy_net:
                policy_params = self.quant_policy_net(task_embedding)
                scale = policy_params[:, 1].mean()
                zero_point = policy_params[:, 2].mean()
            else:
                scale, zero_point = 1.0, 0.0  # Defaults
            
            q_x = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
            x = q_x.dequantize()
        
        if self.config.ctm_use_qat and self.training:
            output = self.dequant(output)

        return output, final_h, confidence

    def enable_quant_aware_training(self, enable: bool = True) -> None:
        self.quant_aware = enable
