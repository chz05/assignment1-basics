import torch
import einops

class RotaryPositionEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct
        the RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
        # Example: d_k=8, max_seq_len=4, theta=10000
        
        # torch.arange(0, d_k, 2) -> [0, 2, 4, 6] shape: (d_k/2,) = (4,)
        indices = torch.arange(0, d_k, 2, device=device)
        
        # freqs[i] = 1 / (theta^(indices[i]/d_k))
        # -> [1/10000^0, 1/10000^0.25, 1/10000^0.5, 1/10000^0.75]
        # shape: (d_k/2,) = (4,)
        freqs = 1.0 / (theta ** (indices / d_k))
        
        # torch.arange(max_seq_len) -> [0, 1, 2, 3] shape: (max_seq_len,) = (4,)
        pos = torch.arange(max_seq_len, device=device)
        
        # einsum outer product: (seq,) x (d,) -> (seq, d)
        # angles[pos_i, freq_j] = pos_i * freqs[j]
        # shape: (max_seq_len, d_k/2) = (4, 4)
        angles = einops.einsum(pos, freqs, 'seq, d -> seq d')
        
        # torch.cos/sin element-wise, same shape: (max_seq_len, d_k/2)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        positions of x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors
        along the sequence dimension
        """
        # Example: x shape (batch=2, seq=4, d_k=8), token_positions shape (seq=4,)
        
        # self.cos shape: (max_seq_len, d_k/2) = (4, 4)
        # token_positions: [0, 1, 2, 3] selects rows from self.cos
        # cos shape after indexing: (..., seq, d_k/2) = (4, 4)
        # Broadcasting will handle batch dims automatically
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        
        # x: (batch, seq, d_k) = (2, 4, 8)
        # Reshape to pair up adjacent elements: (2, 4, 8) -> (2, 4, 4, 2)
        # So x[..., 0] and x[..., 1] become a pair, x[..., 2] and x[..., 3] become a pair, etc.
        x_pairs = einops.rearrange(x, '... seq (d two) -> ... seq d two', two=2)
        
        # x_pairs[..., 0] = even indices: x[..., 0], x[..., 2], x[..., 4], x[..., 6]
        # x_pairs[..., 1] = odd indices:  x[..., 1], x[..., 3], x[..., 5], x[..., 7]
        # Both shapes: (batch, seq, d_k/2) = (2, 4, 4)
        x_even, x_odd = x_pairs[..., 0], x_pairs[..., 1]
        
        # Apply 2D rotation to each (even, odd) pair:
        # [cos -sin] [x_even]   [x_even*cos - x_odd*sin]
        # [sin  cos] [x_odd ] = [x_even*sin + x_odd*cos]
        # Each result shape: (batch, seq, d_k/2) = (2, 4, 4)
        # torch.stack along dim=-1 creates new dim: (2, 4, 4, 2)
        x_rotated = torch.stack([
            x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos
        ], dim=-1)
        
        # Flatten last two dims to interleave: (2, 4, 4, 2) -> (2, 4, 8)
        # Result: [rot_even_0, rot_odd_0, rot_even_1, rot_odd_1, ...]
        return einops.rearrange(x_rotated, '... seq d two -> ... seq (d two)')
