import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MambaBlock(nn.Module):
    """
    A self-contained Mamba block implementation.
    This implementation is based on the principles of State Space Models (SSMs)
    and the selective scan mechanism (S6) from the Mamba paper.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model

        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)

        # Convolutional branch
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # Selective scan (S6) parameters
        # x_proj projects the input to a space for delta, B, C
        self.x_proj = nn.Linear(self.d_inner, self.d_state + self.d_state * 2)

        # dt_proj projects from the x_proj output's delta part to d_inner
        self.dt_proj = nn.Linear(self.d_state, self.d_inner)


        # A parameter (discretization parameter)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Mamba block.
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        b, l, d = x.shape
        
        # 1. Input projection and split
        x_and_res = self.in_proj(x)
        x_in, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        
        # 2. Convolutional branch
        x_in_permuted = x_in.permute(0, 2, 1) # (b, d_inner, l)
        x_conv = self.conv1d(x_in_permuted)
        x_conv = x_conv[:, :, :l] # Trim padding
        x_conv_permuted = x_conv.permute(0, 2, 1) # (b, l, d_inner)
        
        # Apply SiLU activation
        x_conv_act = F.silu(x_conv_permuted)
        
        # 3. Selective Scan (S6)
        y = self.ssm(x_conv_act)
        
        # 4. Residual connection and output projection
        output = y * F.silu(res)
        output = self.out_proj(output)
        
        return output

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        The selective scan mechanism (S6).
        Args:
            x: Input tensor of shape (batch, seq_len, d_inner)
        Returns:
            Output tensor of shape (batch, seq_len, d_inner)
        """
        b, l, d_inner = x.shape
        
        # Project x to get delta, B, and C
        # x_proj_out shape: (b, l, d_state * 3)
        x_proj_out = self.x_proj(x)
        delta, B, C = x_proj_out.split(split_size=[self.d_state, self.d_state, self.d_state], dim=-1)

        # Softplus activation for delta to ensure positivity
        delta = F.softplus(self.dt_proj(delta)) # (b, l, d_inner)
        
        # Discretize A and B
        # A is A_log, B and C are from x_proj
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)
        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (b, l, d_inner, d_state)
        deltaB_x = torch.einsum('bln,bld,bld->bldn', delta, B, x)

        # Initialize hidden state
        h = torch.zeros(b, d_inner, self.d_state, device=x.device)
        
        # Iterate over sequence length
        ys = []
        for i in range(l):
            h = deltaA[:, i] * h + deltaB_x[:, i].unsqueeze(-1)
            y = torch.einsum('bdn,bld->bd', h, C.unsqueeze(2))
            ys.append(y)
        
        y = torch.stack(ys, dim=1) # (b, l, d_inner)

        return y + x * self.D

def ssd(a, B, C, x, chunk_size=256, h_init=None):
    b_size, seq_len, p = x.shape
    n = B.shape[-1]
    if h_init is None:
        h = torch.zeros(b_size, p, n, device=x.device)
    else:
        h = h_init
    y_list = []
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    for ck in range(num_chunks):
        start = ck * chunk_size
        end = min(start + chunk_size, seq_len)
        tc = end - start
        a_ck = a[:, start:end]
        B_ck = B[:, start:end]
        C_ck = C[:, start:end]
        x_ck = x[:, start:end]
        # local_y
        log_a_ck = torch.log(a_ck + 1e-6)
        cum = torch.cumsum(log_a_ck, dim=1)  # (b, tc)
        log_L = cum.unsqueeze(2) - cum.unsqueeze(1)  # (b, tc, tc)
        L = torch.exp(log_L)
        mask = torch.tril(torch.ones((tc, tc), device=x.device), diagonal=0).unsqueeze(0)
        L = L * mask
        inner = torch.bmm(C_ck, B_ck.transpose(1, 2))  # (b tc tc)
        M = inner * L
        local_y = torch.bmm(M, x_ck)  # (b tc p)
        # contrib
        cum_prod = torch.exp(torch.cumsum(log_a_ck, dim=1))  # (b, tc)
        contrib = torch.bmm(C_ck, h.transpose(1, 2))  # (b tc p)
        contrib = cum_prod.unsqueeze(2) * contrib
        y_ck = local_y + contrib
        y_list.append(y_ck)
        # update h
        for s in range(tc):
            a_s = a_ck[:, s].view(-1, 1, 1)  # Reshape for broadcasting
            # Update h in-place to avoid large intermediate tensor allocation
            h.mul_(a_s)
            h.addbmm_(x_ck[:, s].unsqueeze(2), B_ck[:, s].unsqueeze(1))
    y = torch.cat(y_list, dim=1)
    return y

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
