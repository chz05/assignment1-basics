import torch
import einops
from cs336_basics.functions import scaled_dot_product_attention
from cs336_basics.rope import RotaryPositionEmbedding


class MultiHeadSelfAttention(torch.nn.Module):
    """Multi-head self-attention WITHOUT RoPE."""

    def __init__(
        self, 
        d_model: int, 
        num_heads: int,
        q_proj_weight: torch.Tensor,  # (num_heads * d_k, d_model)
        k_proj_weight: torch.Tensor,  # (num_heads * d_k, d_model)
        v_proj_weight: torch.Tensor,  # (num_heads * d_v, d_model)
        o_proj_weight: torch.Tensor,  # (d_model, num_heads * d_v)
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        # Store weights (shape: out_dim x in_dim)
        self.q_proj_weight = q_proj_weight
        self.k_proj_weight = k_proj_weight
        self.v_proj_weight = v_proj_weight
        self.o_proj_weight = o_proj_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Project Q, K, V: (batch, seq, d_model) @ (d_model, num_heads*d_k).T -> (batch, seq, num_heads*d_k)
        q = einops.einsum(x, self.q_proj_weight, 'b s d, out d -> b s out')
        k = einops.einsum(x, self.k_proj_weight, 'b s d, out d -> b s out')
        v = einops.einsum(x, self.v_proj_weight, 'b s d, out d -> b s out')
        
        # Reshape to (batch, num_heads, seq, d_k)
        q = einops.rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = einops.rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = einops.rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)
        
        # Causal mask: True = attend (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        
        # Scaled dot-product attention
        attn_out = scaled_dot_product_attention(q, k, v, mask)
        
        # Reshape back: (batch, num_heads, seq, d_v) -> (batch, seq, num_heads*d_v)
        attn_out = einops.rearrange(attn_out, 'b h s d -> b s (h d)')
        
        # Output projection: (batch, seq, num_heads*d_v) @ (num_heads*d_v, d_model).T -> (batch, seq, d_model)
        out = einops.einsum(attn_out, self.o_proj_weight, 'b s hd, d hd -> b s d')
        
        return out


class MultiHeadSelfAttentionWithRoPE(torch.nn.Module):
    """Multi-head self-attention WITH RoPE."""

    def __init__(
        self, 
        d_model: int, 
        num_heads: int,
        max_seq_len: int,
        theta: float,
        q_proj_weight: torch.Tensor,
        k_proj_weight: torch.Tensor,
        v_proj_weight: torch.Tensor,
        o_proj_weight: torch.Tensor,
        device=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        self.q_proj_weight = q_proj_weight
        self.k_proj_weight = k_proj_weight
        self.v_proj_weight = v_proj_weight
        self.o_proj_weight = o_proj_weight
        
        # RoPE embedding
        self.rope = RotaryPositionEmbedding(theta, self.d_k, max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        token_positions: (batch, seq_len) or (seq_len,) - optional, defaults to [0, 1, 2, ...]
        returns: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Default positions if not provided
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        
        # Project Q, K, V
        q = einops.einsum(x, self.q_proj_weight, 'b s d, out d -> b s out')
        k = einops.einsum(x, self.k_proj_weight, 'b s d, out d -> b s out')
        v = einops.einsum(x, self.v_proj_weight, 'b s d, out d -> b s out')
        
        # Reshape to (batch, num_heads, seq, d_k)
        q = einops.rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = einops.rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = einops.rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)
        
        # Apply RoPE to Q and K
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        
        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        
        # Attention
        attn_out = scaled_dot_product_attention(q, k, v, mask)
        
        # Reshape and output projection
        attn_out = einops.rearrange(attn_out, 'b h s d -> b s (h d)')
        out = einops.einsum(attn_out, self.o_proj_weight, 'b s hd, d hd -> b s d')
        
        return out