import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_expert=16, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
        self.num_expert = num_expert
        self.tempure = 100

    def forward(self, flat_input, top_k=2, beta=0.05):
        # # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = flat_input.shape
        
        # Flatten input
        #flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=flat_input.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input)
        q_latent_loss = F.mse_loss(quantized, flat_input.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = flat_input + (quantized - flat_input).detach() 
        #avg_probs = torch.mean(encodings, dim=0)
        #perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        #get top k
        _, gate_top_k_idx = torch.topk(
                distances * (-1), k=top_k, dim=1, largest=True, sorted=False
            )  # [.. x top_k]
        # #hash index
        gate_top_k_idx = gate_top_k_idx % self.num_expert
        # gate_top_k_val = gate_top_k_val.view(-1, top_k)
        # # (BxL) x 1 x top_k

        # gate_score = F.softmax(gate_top_k_val/self.tempure, dim=-1)
        # convert quantized from BHWC -> BCHW
        return loss, quantized.contiguous(), gate_top_k_idx #, (gate_top_k_idx, gate_score) #, perplexity, encodings, encoding_indices, distances
