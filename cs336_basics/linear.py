import torch
import torch.nn as nn
import einops
import math
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None): 
        """
        Construct a linear transformation module. This function should accept the following parameters:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        sigma = math.sqrt(2 / in_features + out_features)
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=sigma,
            a=-3 * sigma,
            b=3 * sigma,
        )
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input
        """
        return einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")