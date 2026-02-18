import torch
from torch import Tensor
from cs336_basics.MHA import MultiHeadSelfAttentionWithRoPE
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, weights: dict[str, Tensor]):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Strip "layer.X." prefix if present
        w = {}
        for k, v in weights.items():
            # Remove "layer.X." prefix (e.g., "layer.0.ln1.weight" -> "ln1.weight")
            parts = k.split(".")
            if len(parts) > 2 and parts[0] == "layer" and parts[1].isdigit():
                w[".".join(parts[2:])] = v
            else:
                w[k] = v
        
        # pre norm
        self.rmsnorm = RMSNorm(d_model)
        self.rmsnorm.load_state_dict({"gain": w["ln1.weight"]})
        
        self.mha = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta, w["attn.q_proj.weight"], w["attn.k_proj.weight"], w["attn.v_proj.weight"], w["attn.output_proj.weight"])
        self.swiglu = SwiGLU(d_model, d_ff)
        self.swiglu.load_state_dict({"w1.weight": w["ffn.w1.weight"], "w2.weight": w["ffn.w2.weight"], "w3.weight": w["ffn.w3.weight"]})
        self.rmsnorm2 = RMSNorm(d_model)
        self.rmsnorm2.load_state_dict({"gain": w["ln2.weight"]})


    def forward(self, x):
        y = x + self.mha(self.rmsnorm(x))
        z = y + self.swiglu(self.rmsnorm2(y))
        return z