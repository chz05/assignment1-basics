import torch
import math
import einops


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Subtract max for numerical stability (any constant works mathematically,
    # but max makes all values ≤ 0, preventing exp() overflow)
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Q: (..., seq_q, d_k)
    K: (..., seq_k, d_k)
    V: (..., seq_k, d_v)
    mask: (..., seq_q, seq_k) - True means ATTEND, False means mask out
    Returns: (..., seq_q, d_v)
    """
    d_k = Q.shape[-1]
    
    # einsum: contract over d_k dimension
    # (..., seq_q, d_k) x (..., seq_k, d_k) -> (..., seq_q, seq_k)
    scores = einops.einsum(Q, K, '... q d, ... k d -> ... q k') / math.sqrt(d_k)
    
    # Apply mask: True means ATTEND, False means mask out (PyTorch convention)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Softmax over keys dimension (last dim)
    attn_weights = softmax(scores, dim=-1)
    
    # einsum: weighted sum over seq_k dimension
    # (..., seq_q, seq_k) x (..., seq_k, d_v) -> (..., seq_q, d_v)
    return einops.einsum(attn_weights, V, '... q k, ... k v -> ... q v')



def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss.
    logits: (..., vocab_size) - raw unnormalized scores
    targets: (...,) - integer class indices
    Returns: scalar mean loss
    """
    # Numerical stability: subtract max before exp
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logits
    
    # log_sum_exp = max + log(sum(exp(shifted)))
    log_sum_exp = max_logits.squeeze(-1) + torch.log(torch.sum(torch.exp(shifted), dim=-1))
    
    # Get logit for target class using gather (handles arbitrary batch dims)
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # Cross entropy: -log(softmax[target]) = log_sum_exp - target_logit
    loss = log_sum_exp - target_logits
    
    return loss.mean()
