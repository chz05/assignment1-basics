import torch
import torch.nn as nn

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None): 
        """
        Construct an embedding module. This function should accept the following parameters:
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.embedding_matrix = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.embedding_matrix, mean=0.0, std=1, a=-3, b=3)
        self.device = device
        self.dtype = dtype


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor: 
        """
        Lookup the embedding vectors for the given token IDs
        """
        return self.embedding_matrix[token_ids]
