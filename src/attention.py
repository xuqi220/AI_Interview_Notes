import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

@dataclass
class ModelConfig:
    batch_size: int = 6
    block_size: int = 8
    n_embd: int = 12
    n_head: int = 2

class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_w = nn.Linear(config.n_embd, 3*config.n_embd)
        self.proj_w = nn.Linear(config.n_embd, config.n_embd)
    
    def forward(self, x):
        # 计算Q，K，V
        qkv = self.attn_w(x) # [B, T, C]->[B, T, 3*C]
        q,k,v = qkv.split(x.shape[-1], dim=2) # [B, T, 3*C]->3*[B, T, C]
        # 计算相关性分数
        atten_score = q@k.transpose(-2,-1)/(1/math.sqrt(q.shape[-1]))#[B, T, T]
        # 归一化相关性分数
        atten_score = F.softmax(atten_score, dim=-1)#[B, T, T]
        # 加权求和
        out = atten_score@v # [B, T, C]
        return self.proj_w(out)
        
class MHAttention(nn.Module):
    def __init__(self, config:ModelConfig):
        super().__init__()
        assert config.n_embd%config.n_head == 0,"参数设置错误"
        self.config = config
        self.attn_w = nn.Linear(config.n_embd, 3*config.n_embd)
        self.proj_w = nn.Linear(config.n_embd, config.n_embd)
        
    def forward(self,x):
        B, T, C = x.shape
        # 获取Q，K，V矩阵
        qkv = self.attn_w(x) # [B, T, C]->[B,T,3*C]
        q, k, v = qkv.split(self.config.n_embd, dim=-1)
        # [B, T, C]->[B, n_head, T, C/n_head]
        q = q.view(B, T, self.config.n_head, C//self.config.n_head).transpose(2,1)
        k = k.view(B, T, self.config.n_head, C//self.config.n_head).transpose(2,1)
        v = v.view(B, T, self.config.n_head, C//self.config.n_head).transpose(2,1)
        # 计算相关性分数
        attn_score = q@k.transpose(-2,-1)/(1/math.sqrt(C//self.config.n_head)) # [B, n_head, T, T]
        # 相关性分数归一化
        attn_score = F.softmax(attn_score, dim=-1)
        # 对v加权求和
        out = attn_score@v # [B, n_head, T, C/n_head]
        out = out.transpose(1,2).contiguous().view(B, T, C)
        return self.proj_w(out)
    
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
        # 计算相关性分数
        attn_score = q@k.transpose(-2,-1) # [B, n_head, T, T]
        # mask for casual attention
        attn_score = attn_score.masked_fill(mask=self.bias[:,:,:T,:T]==0, value=float("-inf"))
        # 归一化相关性分数
        attn_score = F.softmax(attn_score, dim=-1)
        # 加权求和
        out = attn_score@v # [B, n_head, T, C//n_head]
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
    
        
        