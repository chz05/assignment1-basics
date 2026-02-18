import torch
from torch import Tensor
from cs336_basics.embedding import Embedding
from cs336_basics.transformerblock import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear


class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, weights: dict[str, Tensor]):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.weights = weights
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.token_embeddings.load_state_dict({"embedding_matrix": weights["token_embeddings.weight"]})
        self.transformer_blocks = torch.nn.ModuleList()
        for i in range(num_layers):
            weight = {f"attn.q_proj.weight": weights[f"layers.{i}.attn.q_proj.weight"], f"attn.k_proj.weight": weights[f"layers.{i}.attn.k_proj.weight"], f"attn.v_proj.weight": weights[f"layers.{i}.attn.v_proj.weight"], f"attn.output_proj.weight": weights[f"layers.{i}.attn.output_proj.weight"], f"ffn.w1.weight": weights[f"layers.{i}.ffn.w1.weight"], f"ffn.w2.weight": weights[f"layers.{i}.ffn.w2.weight"], f"ffn.w3.weight": weights[f"layers.{i}.ffn.w3.weight"], f"ln1.weight": weights[f"layers.{i}.ln1.weight"], f"ln2.weight": weights[f"layers.{i}.ln2.weight"]}
            self.transformer_blocks.append(TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, 
            weight))
        self.final_norm = RMSNorm(d_model)
        self.final_norm.load_state_dict({"gain": weights["ln_final.weight"]})
        self.linear = Linear(d_model, vocab_size)
        self.linear.load_state_dict({"weight": weights["lm_head.weight"]})

    def forward(self, x):
        x = self.token_embeddings(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.final_norm(x)
        x = self.linear(x)
        return x