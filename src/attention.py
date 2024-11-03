import torch
import torch.nn as nn
from dataclasses import dataclass
import math

@dataclass
class ModelConfig:
    batch_size = 6
    block_size = 8
    n_embd: int = 12


class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_w = nn.Linear(config.n_embd, 3*config.n_embd)
    
    def forward(self, x):
        # 计算Q，K，V
        qkv = self.attn_w(x) # [B, T, C]->[B, T, 3*C]
        q,k,v = qkv.split(x.shape[-1], dim=2) # [B, T, 3*C]->3*[B, T, C]
        # 计算相关性分数
        atten_score = q@k.transpose(-2,-1)/(1/math.sqrt(q.shape[-1]))#[B, T, T]
        # 归一化相关性分数
        atten_score = torch.softmax(atten_score,dim=-1)#[B, T, T]
        # 加权求和
        out = atten_score@v # [B, T, C]
        return out
        
class MHAttention(nn.Module):
    pass


if __name__=="__main__":
    config = ModelConfig()
    att_net = SelfAttention(config)
    sample = torch.randn((config.batch_size, config.block_size, config.n_embd))
    out = att_net(sample)
    print(out.is_contiguous())
    # print(out)
    
        
        