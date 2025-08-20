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
    
class HierTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, batch_first=True, norm_first=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.transformer_level_A = nn.TransformerEncoderLayer(
            d_model, num_heads, batch_first=batch_first, norm_first=norm_first
        )
        self.transformer_level_B = nn.TransformerEncoderLayer(
            d_model, num_heads, batch_first=batch_first, norm_first=norm_first
        )

    def forward(self, src, src_key_padding_mask, attn_mask_A, attn_mask_B):
        """
        src:                [B, L, d] (batch_first=True)
        attn_mask_A/B:      [B*num_heads, L, L] 或 None；True=禁止，False=允许
        src_key_padding_mask: [B, L]；True=PAD
        """
        B, L, _ = src.shape
        H = self.num_heads
        device = src.device

        # ---- Phase A ----
        yA = self.transformer_level_A(
            src,
            src_mask=attn_mask_A,                      # 注意参数名是 src_mask
            src_key_padding_mask=src_key_padding_mask
        )                                             # [B, L, d]

        if attn_mask_A is not None and attn_mask_A.dim() == 3:
            rows_A = self._rows_from_attn_mask(attn_mask_A, B, H)  # [B, L] True=应更新
        else:
            rows_A = torch.ones(B, L, dtype=torch.bool, device=device)

        x = self._blend_update(src, yA, rows_A)      # 只写回允许的行

        # ---- Phase B ----
        yB = self.transformer_level_B(
            x,
            src_mask=attn_mask_B,
            src_key_padding_mask=src_key_padding_mask
        )

        if attn_mask_B is not None and attn_mask_B.dim() == 3:
            rows_B = self._rows_from_attn_mask(attn_mask_B, B, H)
        else:
            rows_B = torch.ones(B, L, dtype=torch.bool, device=device)

        x = self._blend_update(x, yB, rows_B)

        return x

    @staticmethod
    def _blend_update(x_old: torch.Tensor, x_new: torch.Tensor, update_rows: torch.BoolTensor):
        """
        只在 update_rows=True 的行用 x_new 覆盖；其余行保留 x_old
        x_old, x_new: [B, L, d]
        update_rows:  [B, L] (True=更新)
        """
        mask = update_rows.unsqueeze(-1)              # [B, L, 1]
        return torch.where(mask, x_new, x_old)

    @staticmethod
    def _rows_from_attn_mask(attn_mask: torch.Tensor, B: int, H: int):
        """
        从 [B*H, L, L] 的 attn_mask 推断“允许作为 Query 的行”（即该行存在至少1个非自身的未屏蔽列）。
        返回 [B, L] 的 bool：True 表示该行会被本次前向更新（据此回写）。
        约定：attn_mask==True 为“禁止”，False 为“允许”。
        """
        BH, L, _ = attn_mask.shape
        assert BH == B * H, f"attn_mask 第一维应为 B*num_heads, got {BH} vs {B}*{H}"
        m = attn_mask.view(B, H, L, L)  # [B, H, L, L]
        
        # 创建对角线掩码，忽略对角线的影响
        diag_mask = torch.eye(L, dtype=torch.bool, device=attn_mask.device)  # [L, L]
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(0).expand(B, H, L, L)  # [B, H, L, L]
        
        # 将对角线位置设置为 True（屏蔽），以检查非对角线的 False
        m_non_diag = m | diag_mask  # [B, H, L, L]
        
        # 检查每行（Query）是否全屏蔽（忽略对角线）
        row_all_banned = m_non_diag.all(dim=-1)  # [B, H, L]
        
        # 如果某行在任一 head 中有非自身的未屏蔽列，则需要更新
        row_updatable = (~row_all_banned).any(dim=1)  # [B, L]
        
        return row_updatable
