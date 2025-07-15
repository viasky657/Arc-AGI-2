import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape

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
        new_b = b * self.n_heads
        a_flat = a.reshape(new_b, l)
        B_flat = B.reshape(new_b, l, self.d_state)
        C_flat = C.reshape(new_b, l, self.d_state)
        x_flat = input_x.reshape(new_b, l, self.d_head)

        y_flat = ssd(a_flat, B_flat, C_flat, x_flat, self.chunk_size)

        y = y_flat.view(b, l, self.n_heads, self.d_head)
        y = y.reshape(b, l, self.d_inner)

        output = y * F.silu(res)
        output = self.out_proj(output)
        return output
