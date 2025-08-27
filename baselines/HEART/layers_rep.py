import math
import torch
import torch.nn as nn

from torch_geometric.utils import softmax
from torch_scatter import scatter_sum


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        return self.fc2(self.dropout(gelu(self.fc1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        self.d_k, self.n_heads = config["hidden_size"] // config["num_attention_heads"], config["num_attention_heads"]

        self.W_Q = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.W_K = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.W_V = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.W_output = nn.Linear(config["hidden_size"], config["hidden_size"])

        self.dropout = nn.Dropout(config["attention_probs_dropout_prob"])
    
    def ScaledDotProductAttention(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Q.size(-1)) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = self.dropout(nn.Softmax(dim=-1)(scores))
        context = torch.matmul(attn, V)
        return context, attn

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = self.ScaledDotProductAttention(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k) # context: [batch_size x len_q x n_heads * d_v]
        return self.W_output(context)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.pos_ffn = PoswiseFeedForwardNet(config)

        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.norm_attn = nn.LayerNorm(config["hidden_size"])
        self.norm_ffn = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, self_attn_mask):
        norm_x = self.norm_attn(x)
        x = x + self.dropout(self.self_attn(norm_x, norm_x, norm_x, self_attn_mask))

        norm_x = self.norm_ffn(x)
        x = x + self.dropout(self.pos_ffn(norm_x))
        return x
    
    
class DotAttnConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads=1, n_max_visits=15, temp=1.):
        super(DotAttnConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads, self.temp = n_heads, temp

        self.pos_encoding = nn.Embedding(n_max_visits, in_channels)
        self.W_q = nn.Linear(in_channels, out_channels, bias=False)
        self.W_k = nn.Linear(in_channels, out_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)
        self.W_out = nn.Linear(out_channels, out_channels, bias=False)
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, visit_pos):
        # x: [N, in_channels], edge_index: [2, E]
        N, device = x.size(0), x.device
        isolated_nodes_mask = ~torch.isin(torch.arange(N).to(x.device), edge_index[1].unique())
        isolated_nodes = isolated_nodes_mask.nonzero(as_tuple=False).squeeze()

        pos_encoding = self.pos_encoding(visit_pos)
        h_q, h_k, h_v = self.W_q(x + pos_encoding), self.W_k(x + pos_encoding), self.W_k(x)
        h_q, h_k, h_v = h_q.reshape(N, self.n_heads, -1), h_k.reshape(N, self.n_heads, -1), h_v.reshape(N, self.n_heads, -1)
        
        attn_scores = torch.sum(h_q[edge_index[0]] * h_k[edge_index[1]], dim=-1) / self.temp  # [N_edges, n_heads]
        dst_nodes = torch.cat([edge_index[1] + N*i for i in range(self.n_heads)], dim=0).to(device)
        attn_scores = softmax(attn_scores.reshape(-1), dst_nodes, num_nodes=N * self.n_heads).unsqueeze(dim=-1)  # [N_edges * n_heads, 1]

        # aggregation
        h_v = h_v.permute(1, 0, 2).reshape(N*self.n_heads, -1)
        src_nodes = torch.cat([edge_index[0] + N*i for i in range(self.n_heads)], dim=0).to(device)
        out = scatter_sum(src=h_v[src_nodes] * attn_scores, index=dst_nodes, dim_size=N * self.n_heads, dim=0)
        out = out.reshape(self.n_heads, N, -1).permute(1, 0, 2).reshape(N, -1)

        out = self.W_out(self.ln(out)) + x
        out[isolated_nodes] = x[isolated_nodes]
        return out

