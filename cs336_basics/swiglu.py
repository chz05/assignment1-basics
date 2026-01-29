import torch
from cs336_basics.linear import Linear

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        Construct the SwiGLU module. This function should accept the following parameters:
        d_model: int Hidden dimension of the model
        d_ff: int Hidden dimension of the feedforward layer
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = ((8 * d_model // 3 + 63) // 64) * 64
        self.device = device
        self.dtype = dtype
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        return self.w2(gate * torch.sigmoid(gate) * self.w3(x))