import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
import math

@dataclass
class ModelConfig:
    batch_size: int = 6
    block_size: int = 8
    n_embd: int = 12
    n_head: int = 2

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)   

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


class CasualAttention(nn.Module):
    def __init__(self, config:ModelConfig):
        super().__init__()
        self.config = config
        self.attn_w = nn.Linear(config.n_embd, config.n_embd*3)
        self.proj_w = nn.Linear(config.n_embd, config.n_embd)
        # self.register_buffer( # MASK
        #     "bias", 
        #     torch.tril(torch.ones(self.config.block_size, self.config.block_size)).view(1,1, self.config.block_size,self.config.block_size)
        # )
    def forward(self, x):
        B, T, C = x.shape
        # 获取Q，K，V
        qkv = self.attn_w(x) # [B, T, C]->[B, T, C*3]
        q, k, v = qkv.split(C, dim=-1) # [B, T, C]
        # 多头 [B, n_head, T, C//n_head]
        q = q.view(B, T, self.config.n_head, C//self.config.n_head).transpose(1,2) 
        k = k.view(B, T, self.config.n_head, C//self.config.n_head).transpose(1,2)
        v = v.view(B, T, self.config.n_head, C//self.config.n_head).transpose(1,2)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        # Flash_attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1,2).contiguous().view(B, T, C) # [B, T, C]
        out = self.proj_w(out)
        return out
        
if __name__=="__main__":
    config = ModelConfig()
    att_net = CasualAttention(config)
    sample = torch.randn((config.batch_size, config.block_size, config.n_embd))
    out = att_net(sample)
    print(out.is_contiguous())
    # print(out)
    
        
        