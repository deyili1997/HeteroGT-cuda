import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv

class DiseaseOccHetGNN(nn.Module):
    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d = d_model
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # —— 规范化：按节点类型各自一套 LN —— #
        self.ln_v1 = nn.LayerNorm(d_model)
        self.ln_o1 = nn.LayerNorm(d_model)
        self.ln_v2 = nn.LayerNorm(d_model)
        self.ln_o2 = nn.LayerNorm(d_model)

        # —— 可学习缩放（残差权重），初始化小值避免早期干扰 —— #
        self.alpha_v1 = nn.Parameter(torch.tensor(0.1))
        self.alpha_o1 = nn.Parameter(torch.tensor(0.1))
        self.alpha_v2 = nn.Parameter(torch.tensor(0.1))
        self.alpha_o2 = nn.Parameter(torch.tensor(0.1))

        # 注意：这里用 aggr='sum'，让关系信号不被平均稀释
        self.conv1 = HeteroConv({
            ('visit','contains','occ'):   GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=False),
            ('occ','contained_by','visit'): GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=False),
            ('visit','next','visit'):     GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=True),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('visit','contains','occ'):   GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=False),
            ('occ','contained_by','visit'): GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=False),
            ('visit','next','visit'):     GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=True),
        }, aggr='sum')

        # 末端线性 + 残差（用零初始化保持“近似恒等”）
        self.lin_v = nn.Linear(d_model, d_model)
        self.lin_o = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.lin_v.weight); nn.init.zeros_(self.lin_v.bias)
        nn.init.zeros_(self.lin_o.weight); nn.init.zeros_(self.lin_o.bias)

    def forward(self, hg):
        # x_dict: {'visit': [N_visit, d], 'occ': [N_occ, d]}
        x_v = hg['visit'].x
        x_o = hg['occ'].x

        # ===== Layer 1: 图卷积（sum 聚合）→ 残差 + LN =====
        h1 = self.conv1({'visit': x_v, 'occ': x_o}, hg.edge_index_dict)
        # 残差注入前先丢弃避免过拟合
        dv = self.drop(h1['visit'])
        do = self.drop(h1['occ'])
        # y = LN(x + α * Δx)
        v1 = self.ln_v1(x_v + self.alpha_v1 * dv)
        o1 = self.ln_o1(x_o + self.alpha_o1 * do)

        # ===== Layer 2: 再一层图卷积 → 残差 + LN =====
        h2 = self.conv2({'visit': v1, 'occ': o1}, hg.edge_index_dict)
        dv2 = self.drop(h2['visit'])
        do2 = self.drop(h2['occ'])
        v2 = self.ln_v2(v1 + self.alpha_v2 * dv2)
        o2 = self.ln_o2(o1 + self.alpha_o2 * do2)

        # ===== 末端线性：零初始化，等价“细调残差”，不改变整体尺度期望 =====
        v_out = v2 + self.lin_v(v2)
        o_out = o2 + self.lin_o(o2)

        return {'visit': v_out, 'occ': o_out}
    
# multi-class classification task
class MultiPredictionHead(nn.Module):
    def __init__(self, hidden_size, label_size):
        super(MultiPredictionHead, self).__init__()
        self.cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), 
                nn.ReLU(), 
                nn.Linear(hidden_size, label_size)
            )

    def forward(self, input):
        return self.cls(input)
    
class BinaryPredictionHead(nn.Module):
    def __init__(self, hidden_size):
        super(BinaryPredictionHead, self).__init__()
        self.cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), 
                nn.ReLU(), 
                nn.Linear(hidden_size, 1)
            )
    def forward(self, input):
        return self.cls(input)

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