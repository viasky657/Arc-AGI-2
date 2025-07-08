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
