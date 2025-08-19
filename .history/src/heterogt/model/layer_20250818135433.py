import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout=0.1, dim_feedforward=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=num_heads,
                    batch_first=True
                ),
                "linear1": nn.Linear(d_model, dim_feedforward),
                "linear2": nn.Linear(dim_feedforward, d_model),
                "norm1": nn.LayerNorm(d_model),
                "norm2": nn.LayerNorm(d_model),
                "dropout": nn.Dropout(dropout),
                "dropout1": nn.Dropout(dropout),  # 自注意力 Dropout
                "dropout2": nn.Dropout(dropout)   # FFN Dropout
            }) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu  # 与 nn.TransformerEncoderLayer 默认一致

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        src: (B, S, D) - 批量大小, 序列长度, 嵌入维度
        attn_mask: (B, S, S) 或 (S, S)
        key_padding_mask: (B, S) True=mask掉
        """
        x = src
        for layer in self.layers:
            # === 自注意力 (Pre-LayerNorm) ===
            residual = x
            x = layer["norm1"](x)  # 先归一化
            attn_output, _ = layer["self_attn"](
                x, x, x,
                attn_mask=mask,
                key_padding_mask=src_key_padding_mask
            )
            x = residual + layer["dropout1"](attn_output)  # 残差连接 + Dropout

            # === FFN (Pre-LayerNorm) ===
            residual = x
            x = layer["norm2"](x)  # 先归一化
            x = layer["linear2"](self.activation(layer["linear1"](x)))
            x = residual + layer["dropout2"](x)  # 残差连接 + Dropout

        return x