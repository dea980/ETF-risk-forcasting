import torch
import torch.nn as nn
import math

class LogSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(LogSparseAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scale = math.sqrt(d_model // n_heads)

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        H = self.n_heads
        q = self.q_linear(x).view(B, L, H, D // H).transpose(1, 2)
        k = self.k_linear(x).view(B, L, H, D // H).transpose(1, 2)
        v = self.v_linear(x).view(B, L, H, D // H).transpose(1, 2)

        scores = torch.zeros(B, H, L, L).to(x.device)
        for i in range(L):
            idx = [i - 2**j for j in range(int(math.log2(i+1))+1) if i - 2**j >= 0]
            idx.append(i)
            scores[:, :, i, idx] = (q[:, :, i, :] * k[:, :, idx, :]).sum(dim=-1)

        scores = scores / self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class LogSparseTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = LogSparseAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


class LogSparseTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.encoder = nn.Sequential(
            *[LogSparseTransformerEncoderLayer(d_model, n_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.out_proj(x).squeeze(-1)
