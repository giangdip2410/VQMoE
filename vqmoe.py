import torch
import torch.nn as nn
from custom_layers import FMoE
from linear import FMoELinear
import math
from typing import Optional, Union

import torch
from einops import einsum, rearrange
from torch import Tensor, nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from custom_layers import FMoE
from linear import FMoELinear
from vector_quantize_pytorch import VectorQuantize, ResidualVQ, FSQ, ResidualFSQ, LFQ, ResidualLFQ, LatentQuantize
from balance_vq import BVectorQuantiser


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange


class VectorQuantiser(nn.Module):
    """
    Improved version over vector quantiser, with the dynamic initialisation
    for these unoptimised "dead" points.
    num_embed: number of codebook entry
    embed_dim: dimensionality of codebook entry
    beta: weight for the commitment loss
    distance: distance for looking up the closest code
    anchor: anchor sampled methods
    first_batch: if true, the offline version of our model
    contras_loss: if true, use the contras_loss to further improve the performance
    """
    def __init__(self, num_embed, embed_dim, beta=0.25, distance='l2', 
                 anchor='closest', first_batch=True, contras_loss=True):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

        self.pool = FeaturePool(self.num_embed, self.embed_dim)
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))

    
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        #z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.embed_dim)

        # clculate the distance
        if self.distance == 'l2':
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(self.embedding.weight ** 2, dim=1) + \
                2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(self.embedding.weight, 'n d-> d n'))
        elif self.distance == 'cos':
            # cosine distances from z to embeddings e_j 
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            d = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))

        # encoding
        sort_distance, indices = d.sort(dim=1)
        # look up the closest point for the indices
        encoding_indices = indices[:,-1]
        encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # quantise and unflatten
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z * 0.95 + (z_q - z).detach() * 0.05
        # reshape back to match original input shape
        #z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        min_encodings = encodings

        # online clustered reinitialisation for unoptimized points
        if self.training:
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                # closest sampling
                if self.anchor == 'closest':
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1,:]]
                # feature pool based random sampling
                elif self.anchor == 'random':
                    random_feat = self.pool.query(z_flattened.detach())
                # probabilitical based random sampling
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = torch.exp(-(self.embed_prob*self.num_embed*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/self.num_embed)):,:].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss +=  contra_loss
        #z_q = z_q * 0.05 + z * 0.95
        return loss, z_q, encoding_indices #z_q, loss, (perplexity, min_encodings, encoding_indices)

class FeaturePool():
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """
    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1)/ pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)    
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size: # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, activation=nn.ReLU(), bias=True):
        super().__init__()

        self.l1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.activation = activation
        self.l2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x) -> torch.Tensor:
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        return x

class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, contrib_rate=0.05,num_expert=16):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
        self.contrib_rate = contrib_rate
        self.num_expert = num_expert

    def forward(self, inputs):
        # # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs * (1-self.contrib_rate)  + (quantized - inputs).detach() * self.contrib_rate
        #avg_probs = torch.mean(encodings, dim=0)
        #perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        #quantized = inputs * (1-self.contrib_rate) + quantized * self.contrib_rate
        #convert to expert index by hash index
        #encoding_indices = encoding_indices % self.num_expert
        # convert quantized from BHWC -> BCHW
        return quantized.reshape(input_shape).contiguous(), encoding_indices , loss #, perplexity, encodings, encoding_indices, distances


class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x



class VQMoE(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        top_k=2,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,contrib_rate=0.1,all_gate=None,num_embeddings=512, vq_gate=False, vqtype='base',
        **kwargs
    ):
        super().__init__(num_expert=num_expert, d_model=d_model,top_k=top_k,contrib_rate=contrib_rate, **kwargs)
        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank
        )
        self.mark_parallel_comm(expert_dp_comm)
        # if all_gate is None:
        if vqtype == 'vq_cos':
            self.vq = VectorQuantize(dim=d_model, codebook_size=num_embeddings, use_cosine_sim = True, learnable_codebook =True, ema_update = False) 
        elif vqtype == 'bl_vq':
            self.vq = BVectorQuantiser(num_embed=num_expert, embed_dim=d_model)
        elif vqtype == 'base':
            self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=d_model, contrib_rate=contrib_rate, num_expert=num_expert)
        elif vqtype == 'vq_lowdim':
            self.vq = VectorQuantize(dim=d_model, codebook_size=num_embeddings, codebook_dim=num_expert)
        else:
            self.vq = VectorQuantize(dim=d_model, codebook_size=num_embeddings, use_cosine_sim = True, learnable_codebook =True, ema_update = False) 
  
        self.loss = 0.0
        self.num_expert = num_expert
        self.contrib_rate = contrib_rate
        self.layer_norm = nn.LayerNorm(d_model)
        self.combine = nn.Linear(d_model, 2)
    
 
    def activate_experts(self, inputs, selected_experts):
        results = torch.zeros_like(inputs)
        for current_expert_index in range(self.num_expert):
            token_index = selected_experts == current_expert_index
            # Apply the expert to the selected tokens weighting it by the logits (post-softmax) computed above .
            if token_index.shape[0] > 0:
                results[token_index] =  self.activate_one_experts(
                    inputs[token_index], current_expert_index
                )
        return self.layer_norm(results)
    
    def activate_one_experts(self, moe_inp, i):
        temp_ = moe_inp @ self.experts.htoh4.weight[i].T + self.experts.htoh4.bias[i]
        temp_ = F.relu(temp_)
        temp_ = temp_ @ self.experts.h4toh.weight[i].T + self.experts.h4toh.bias[i]
        return temp_

    def forward(self, inp, all_gate=None):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        #if self.top_k > 1:
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        #gate 
        logits = self.combine(inp)
        output = super().forward(inp)
        #vq output
        quantized_all, encoding_indices_all, commit_loss  = self.vq(inp)
        encoding_indices_all = encoding_indices_all.reshape(-1)
        encoding_indices_all = encoding_indices_all % self.num_expert
        self.loss = commit_loss
        #normalize output
        quantized_inp = self.layer_norm(quantized_all)
        output_vq = self.activate_experts(quantized_inp, encoding_indices_all)
        #get vq input
        #combine 2 input
        #print(inp.shape, output.shape, output_vq.shape)
        output = torch.concat([output.unsqueeze(-1), output_vq.unsqueeze(-1)], dim=-1)
        gates = F.softmax(logits, dim=-1) #self.combine(inp) # t x top_k
        #combine output
        final_output = torch.einsum('bij,bjk->bik', output, gates.unsqueeze(-1)).squeeze(-1)
        return final_output.reshape(original_shape)
        