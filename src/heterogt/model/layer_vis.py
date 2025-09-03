import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv
from collections import defaultdict

class DiseaseOccHetGNN(nn.Module):
    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d = d_model
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self.ln_v1 = nn.LayerNorm(d_model)
        self.ln_o1 = nn.LayerNorm(d_model)
        self.ln_v2 = nn.LayerNorm(d_model)
        self.ln_o2 = nn.LayerNorm(d_model)

        self.alpha_v1 = nn.Parameter(torch.tensor(0.1))
        self.alpha_o1 = nn.Parameter(torch.tensor(0.1))
        self.alpha_v2 = nn.Parameter(torch.tensor(0.1))
        self.alpha_o2 = nn.Parameter(torch.tensor(0.1))

        self.conv1 = HeteroConv({
            ('visit','contains','occ'):      GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=False),
            ('occ','contained_by','visit'):  GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=False),
            ('visit','next','visit'):        GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=True),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('visit','contains','occ'):      GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=False),
            ('occ','contained_by','visit'):  GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=False),
            ('visit','next','visit'):        GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=True),
        }, aggr='sum')

        self.lin_v = nn.Linear(d_model, d_model)
        self.lin_o = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.lin_v.weight); nn.init.zeros_(self.lin_v.bias)
        nn.init.zeros_(self.lin_o.weight); nn.init.zeros_(self.lin_o.bias)

    def forward(self, hg, return_attn: bool = True):
        """
        hg: HeteroData，要求:
            - hg['visit'].x, hg['occ'].x
            - hg.edge_index_dict[(src, rel, dst)]
        return:
            out_dict: {'visit': v_out, 'occ': o_out}
            (可选) attn2: dict keyed by (src,rel,dst) with
                {
                    'edge_index': LongTensor [2, E],
                    'alpha': FloatTensor [E, H],          # 每条边、每个 head 的注意力(softmax 后)
                    'alpha_mean': FloatTensor [E],        # head 平均
                }
        """
        x_v = hg['visit'].x
        x_o = hg['occ'].x

        # ----- 第 1 层：正常跑，不取权重 -----
        h1 = self.conv1({'visit': x_v, 'occ': x_o}, hg.edge_index_dict)
        dv = self.drop(h1['visit'])
        do = self.drop(h1['occ'])
        v1 = self.ln_v1(x_v + self.alpha_v1 * dv)
        o1 = self.ln_o1(x_o + self.alpha_o1 * do)

        # ----- 第 2 层：手动遍历关系，收集注意力 -----
        x_dict_in = {'visit': v1, 'occ': o1}
        edge_index_dict = hg.edge_index_dict

        # 聚合器：和 HeteroConv(aggr='sum') 一致
        out_aggr = defaultdict(list)
        attn2 = {}  # {(src, rel, dst): {'edge_index':..., 'alpha':..., 'alpha_mean':...}}

        for rel, conv in self.conv2.convs.items():
            src, _, dst = rel
            x_src = x_dict_in[src]
            x_dst = x_dict_in[dst]
            edge_index = edge_index_dict[rel]

            # 对单个关系的 GATConv 调用并取注意力
            out_rel, (ei, alpha) = conv(
                (x_src, x_dst), edge_index,
                return_attention_weights=True
            )
            # out 聚合
            out_aggr[dst].append(out_rel)

            # 保存注意力（alpha: [E, H]；按 head 平均得到 [E]）
            alpha_mean = alpha.mean(dim=1) if alpha.dim() == 2 else alpha  # 安全处理
            attn2[rel] = {
                'edge_index': ei,            # [2, E]
                'alpha': alpha,              # [E, H]
                'alpha_mean': alpha_mean,    # [E]
            }

        # sum 聚合到各目标类型
        h2 = {ntype: torch.stack(feats, dim=0).sum(dim=0) for ntype, feats in out_aggr.items()}

        dv2 = self.drop(h2['visit'])
        do2 = self.drop(h2['occ'])
        v2 = self.ln_v2(v1 + self.alpha_v2 * dv2)
        o2 = self.ln_o2(o1 + self.alpha_o2 * do2)

        v_out = v2 + self.lin_v(v2)
        o_out = o2 + self.lin_o(o2)

        out_dict = {'visit': v_out, 'occ': o_out}
        return (out_dict, attn2) if return_attn else out_dict
    
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
    
class AttnProbeMHA(nn.MultiheadAttention):
    """
    A thin wrapper over nn.MultiheadAttention that stores the per-head
    attention probabilities from the last forward pass in `.last_attn_probs`.

    Shapes (batch_first=True):
        - attn_out:    [B, L_q, D]
        - attn_probs:  [B, H, L_q, L_k]   (when average_attn_weights=False)
    """
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, **kwargs)
        self.last_attn_probs = None  # set on every forward

    def forward(self, query, key, value, **kwargs):
        # force returning per-head weights
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        attn_out, attn_probs = super().forward(query, key, value, **kwargs)
        self.last_attn_probs = attn_probs  # [B, H, L_q, L_k] (batch_first=True)
        return attn_out
    
class HierTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, batch_first=True, norm_first=True, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            batch_first=batch_first, norm_first=norm_first, dropout=dropout
        )

        # --- swap in the probing MHA and copy weights ---
        old_attn = self.transformer.self_attn
        probe_attn = AttnProbeMHA(
            embed_dim=d_model, num_heads=num_heads,
            dropout=old_attn.dropout, batch_first=True, bias=old_attn.in_proj_bias is not None
        )
        # copy parameters/buffers
        probe_attn.load_state_dict(old_attn.state_dict(), strict=False)
        self.transformer.self_attn = probe_attn

    def forward(self, x, src_key_padding_mask, attn_mask):
        """
        x:                     [B, L, d]
        attn_mask:             [B*num_heads, L, L]  (bool or float, PyTorch-compatible)
        src_key_padding_mask:  [B, L] (True=PAD)
        """
        B, L, _ = x.shape
        H = self.num_heads

        out = self.transformer(
            src=x,
            src_key_padding_mask=src_key_padding_mask,
            src_mask=attn_mask
        )

        rows_use = self._rows_from_attn_mask(attn_mask, src_key_padding_mask, B, H)
        x = self._blend_update(x, out, rows_use)

        # --- pull the attention probabilities from the instrumented self_attn ---
        attn_probs = self.transformer.self_attn.last_attn_probs  # [B, H, L, L]
        # NOTE: attn_probs already respect attn_mask & key_padding_mask (masked positions ~0 after softmax)
        attn_mean = attn_probs.mean(dim=1)  # [B, L, L]
        return x, attn_mean  # return the matrix for visualization

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
    def _rows_from_attn_mask(attn_mask: torch.Tensor, src_key_padding_mask: torch.Tensor, B: int, H: int):
        """
        从 [B*H, L, L] 的 attn_mask 推断“允许作为 Query 的行”(即该行存在至少1个非自身的未屏蔽列, 
        且该行不是填充 token)。返回 [B, L] 的 bool: True 表示该行会被本次前向更新。
        约定:attn_mask==True 为“禁止”, False 为“允许”, src_key_padding_mask==True 为填充 token。
        
        参数：
            attn_mask: [B*H, L, L], torch.bool, 注意力掩码
            src_key_padding_mask: [B, L], torch.bool, 填充掩码, True 表示填充
            B: batch_size
            H: num_heads
        返回：
            row_updatable: [B, L], torch.bool, True 表示该 token 需要更新
        """
        BH, L, _ = attn_mask.shape
        assert BH == B * H, f"attn_mask 第一维应为 B*num_heads, got {BH} vs {B}*{H}"
        assert src_key_padding_mask.shape == (B, L), \
            f"src_key_padding_mask 形状应为 [B, L], got {src_key_padding_mask.shape}"
        assert src_key_padding_mask.dtype == torch.bool, "src_key_padding_mask 必须是 torch.bool 类型"
        assert src_key_padding_mask.device == attn_mask.device, \
            f"设备不匹配: src_key_padding_mask 在 {src_key_padding_mask.device}, attn_mask 在 {attn_mask.device}"

        m = attn_mask.view(B, H, L, L)  # [B, H, L, L]
        
        # 创建对角线掩码，忽略对角线的影响
        diag_mask = torch.eye(L, dtype=torch.bool, device=attn_mask.device)  # [L, L]
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(0).expand(B, H, L, L)  # [B, H, L, L]
        
        # 将对角线位置设置为 True（屏蔽），以检查非对角线的 False，因为我们原来对角线最后设置的是False
        m_non_diag = m | diag_mask  # [B, H, L, L]
        
        # 考虑 src_key_padding_mask，屏蔽填充 token 对应的 Key 位置
        padding_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, L, 1]
        m_non_diag = m_non_diag | padding_mask  # [B, H, L, L]，填充 token 的列全为 True（屏蔽）

        # 检查每行（Query）是否全屏蔽（忽略对角线和填充 token）
        row_all_banned = m_non_diag.all(dim=-1)  # [B, H, L]
        
        # 如果某行在任一 head 中有非自身的未屏蔽列（非填充 token），则需要更新
        row_updatable = (~row_all_banned).any(dim=1)  # [B, L]
        
        # 排除填充 token：填充 token (src_key_padding_mask == True) 不应更新
        row_updatable = row_updatable & (~src_key_padding_mask)  # [B, L]
        return row_updatable
    
class CLSQueryMHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_token_type: list, dropout: float = 0.0, 
                 use_raw_value_agg: bool = True, fallback_to_cls: bool = True):
        super().__init__()
        self.d_model = d_model
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.use_raw_value_agg = use_raw_value_agg
        self.fallback_to_cls = fallback_to_cls
        self.drop = nn.Dropout(dropout)
        self.attn_token_types = attn_token_type

    def forward(self, x: torch.Tensor, token_type: torch.Tensor):
        """
        x: [B, L, D]
        token_type: [B, L] (long/int)
        return:
            out: [B, 2D]  = concat([CLS], agg)
            attn_probs: [B, H, 1, L]  方便调试（每头的注意力）
        """
        B, L, D = x.shape
        assert token_type.shape == (B, L)
        assert D == self.d_model

        # 1) CLS 作为单查询
        cls = x[:, 0, :]                 # [B, D]
        q   = cls.unsqueeze(1)           # [B, 1, D]
        k   = x                          # [B, L, D]
        v   = x                          # [B, L, D]

        # 2) 构造 key_padding_mask：True=忽略（屏蔽）
        allowed = torch.zeros_like(token_type, dtype=torch.bool)
        for t in self.attn_token_types:
            allowed |= (token_type == t)
        kv_mask = ~allowed   # 非 {attn_token_types} 位置屏蔽
        # 防止整行全 True（即没有 {6,7}）导致 softmax NaN：临时放开 CLS 位
        no_kv = kv_mask.all(dim=1)       # [B]
        if no_kv.any():
            kv_mask = kv_mask.clone()
            kv_mask[no_kv, 0] = False    # 避免 NaN；之后会覆盖聚合结果

        # 3) 多头注意力（需要权重；不按头平均）
        attn_out, attn_probs = self.mha(
            q, k, v,
            key_padding_mask=kv_mask,           # [B, L]；True=忽略
            need_weights=True,
            average_attn_weights=False          # -> [B, H, 1, L]
        )  # attn_out: [B, 1, D]

        # 4) 由注意力得到聚合向量
        if self.use_raw_value_agg:
            # 在“输入空间”用权重显式加权得到均值
            w = attn_probs.mean(dim=1)         # [B, 1, L] 按头平均
            # 置零被屏蔽位置，子集重归一化（避免数值泄漏到非 {6,7}）
            w = w.masked_fill(kv_mask.unsqueeze(1), 0.0)  # [B, 1, L]
            denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # [B,1,1]
            w = w / denom
            agg = torch.bmm(w.reshape(B, 1, L), x).squeeze(1)      # [B, D]
        else:
            # 直接使用 MHA 的输出（已在 value 投影+out_proj 空间）
            agg = attn_out.squeeze(1)   # [B, D]

        # 5) 无 {6,7} 的样本回退策略
        if no_kv.any():
            if self.fallback_to_cls:
                agg = agg.clone()
                agg[no_kv] = cls[no_kv]    # 回退为 CLS
            else:
                agg = agg.clone()
                agg[no_kv] = 0.0           # 回退为零向量
        assert agg.shape == (B, D), "CLS output shape mismatch"
        return agg, w  # [B, D]
