import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch as HeteroBatch
from heterogt.model.layer import DiseaseOccHetGNN
from heterogt.model.layer import BinaryPredictionHead, MultiPredictionHead

class HeteroGT(nn.Module):
    def __init__(self, tokenizer, d_model, num_heads, layer_types, max_num_adms, device, task, label_vocab_size):
        super(HeteroGT, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.max_num_adms = max_num_adms
        self.global_vocab_size = len(self.tokenizer.vocab.word2id)
        self.n_type = 4 # diag, med, lab, pro
        self.d_model = d_model
        self.num_attn_heads = num_heads
        self.layer_types = layer_types
        self.seq_pad_id = tokenizer.convert_tokens_to_ids(["[PAD]"], voc_type="all")[0] #0
        self.type_pad_id = 0
        self.adm_pad_id = 0
        self.age_pad_id = tokenizer.convert_tokens_to_ids(["[PAD]"], voc_type="all")[0] #0
        self.node_type_id_dict = {'diag': 1, 'med': 2, 'lab': 3, 'pro': 4, 'group': 5, 'visit': 6}
        self.graph_node_types = ['diag']
        self.forbid_map = {-1: [], 1: [5, 6], 2: [5, 6], 3: [5, 6], 4: [5, 6], 
                           5: [-1, 1, 5, 6], 6: [-1, 1, 5, 6]}  # attention forbid mask, -1 is for the task token

        # embedding layers
        self.token_emb = nn.Embedding(self.global_vocab_size, d_model, padding_idx=self.seq_pad_id) # already contains [PAD], will also be used for age_gender
        self.type_emb = nn.Embedding(self.n_type + 3, d_model, padding_idx=self.type_pad_id) # + 3 for [PAD], group and visit
        self.adm_index_emb = nn.Embedding(self.max_num_adms + 2, d_model, padding_idx=self.adm_pad_id) # +2 for pad and group. Group is using max_num_adms + 1 as adm index
        self.task_emb = nn.Embedding(5, d_model, padding_idx=None)  # 5 task in total, task embedding
        
        # stack together
        self.stack_layers = nn.ModuleList(self.make_gnn_layer() if layer_type == 'gnn' else self.make_tf_layer()
            for layer_type in self.layer_types
        )
        # prediction head
        if task in ["death", "stay", "readmission"]:
            self.cls_head = BinaryPredictionHead(self.d_model)
        else:
            self.cls_head = MultiPredictionHead(self.d_model, label_vocab_size)

    def make_tf_layer(self):
        assert self.d_model % self.num_attn_heads == 0, "Invalid model and attention head dimensions"
        layer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_attn_heads, batch_first=True, norm_first=True)
        tf_wrapper = nn.TransformerEncoder(layer_layer, num_layers=1, enable_nested_tensor=False)
        return tf_wrapper

    def make_gnn_layer(self):
        return DiseaseOccHetGNN(d_model=self.d_model, heads=self.num_attn_heads)
    
    def forward(self, input_ids, token_types, adm_index, age_ids, diag_code_group_dicts, task_id):
        """Forward pass for the model.

        Args:
            input_ids (Tensor): Input token IDs. Shape of [B, L]
            token_types (Tensor): Token type IDs. Shape of [B, L]
            adm_index (Tensor): Admission index IDs. Shape of [B, L]
            age_ids (Tensor): Age IDs. Shape of [B, V]
            diag_code_group_dicts (list): B dictionaries mapping group code idx to their corresponding diag code idx.
            task_id (Tensor): Task ID. Shape of [1]

        Returns:
            Tensor: Output logits. Shape of [B, label_size]
        """
        B, L = input_ids.shape
        V = age_ids.shape[1]
        num_visits = (age_ids != self.age_pad_id).sum(dim=1) # [B], number of visits for each patient in the batch
        
        task_id = torch.full((B,), task_id, dtype=torch.long, device=self.device) # [1] -> [B]
        # 基础表示
        task_id_embed = self.task_emb(task_id).unsqueeze(1) # [B, 1, d]
        seq_embed = self.token_emb(input_ids)  # [B, L, d]
        visit_embed = self.token_emb(age_ids) # [B, V, d]

        # run through layers
        for i, layer_type in enumerate(self.layer_types):
            if layer_type == 'gnn': # purpose is just to update visit_embed
                seq_embed_det = seq_embed.detach()
                visit_embed_det = visit_embed.detach()
                hg_batch = self.build_graph_batch(seq_embed_det, token_types, self.graph_node_types, visit_embed_det, adm_index) # num_visits is a 1d tensor of [B]
                gnn_out = self.stack_layers[i](hg_batch)['visit']  # extract virtual visit node representations
                visit_embed = self.process_gnn_out(gnn_out, num_visits, V) # [B, V, d]
            elif layer_type == 'tf':
                x, src_key_padding_mask, attn_mask = self.prepare_tf_input(task_id_embed, seq_embed, visit_embed, i, input_ids, adm_index, token_types, num_visits, diag_code_group_dicts)
                h = self.stack_layers[i](src=x, src_key_padding_mask=src_key_padding_mask, mask=attn_mask) # [B, 1+L+V, d]
                task_id_embed, seq_embed, visit_embed = self.process_tf_out(h, L, V) # # [B, 1, d], [B, L, d], [B, V, d]
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        logits = self.cls_head(task_id_embed.squeeze())  # [B, label_size]
        return logits

    def build_graph_batch(self, seq_embed, token_types, graph_node_types, visit_embed, adm_index):
        """Build a batch of heterogeneous graphs from the input sequences.

        Args:
            seq_embed (Tensor): Sequence embeddings. Shape of [B, L, d]
            token_types (Tensor): Token type IDs. Shape of [B, L]
            graph_node_types: a list controls what types of tokens are connected to the virtual visit nodes. e.g. ['diag']
            visit_embed (Tensor): Visit embeddings. Shape of [B, V, d]
        Returns:
            A batch of heterogeneous graphs.
        """
        B, L = seq_embed.shape[0], seq_embed.shape[1]
        V = visit_embed.shape[1]
        graph_node_type_ids = [self.node_type_id_dict[t] for t in graph_node_types]
        graphs = [] # contains heterogeneous graphs for each patient
        for p in range(B):
            hg_p = self.build_patient_graph(seq_embed[p], token_types[p], visit_embed[p], adm_index[p], graph_node_type_ids)
            graphs.append(hg_p)
        hg_batch = HeteroBatch.from_data_list(graphs).to(self.device)
        return hg_batch

    def build_patient_graph(self, seq_embed_p, token_types_p, visit_embed_p, adm_index_p, graph_node_type_ids):
        """Build a heterogeneous graph for a single patient.

        Args:
            seq_embed_p (Tensor): Sequence embeddings for patient p. Shape [L, d]
            token_types_p (Tensor): Token type IDs for patient p. Shape [L]
            visit_embed_p (Tensor): Visit embeddings for patient p. Shape [V, d]
            graph_node_type_ids (list): List of graph node type IDs that the graph uses.
            adm_index_p (Tensor): Admission index for patient p. Shape [L]

        Returns:
            A heterogeneous graph for patient p.
        """
        # temporal, trim off group code
        seq_embed_p = seq_embed_p[token_types_p != self.node_type_id_dict['group'], :]
        adm_index_p = adm_index_p[token_types_p != self.node_type_id_dict['group']]
        token_types_p = token_types_p[token_types_p != self.node_type_id_dict['group']]
        
        hg = HeteroData()
        occ_mask = torch.isin(token_types_p, torch.tensor(graph_node_type_ids, device=token_types_p.device)) # [L], a mask for the token types needed in the graph
        occ_pos = torch.nonzero(occ_mask, as_tuple=False).view(-1) # [L], seq position index for the token types needed in the graph
        num_occ = occ_pos.numel() # int, number of occurrences of the token types needed in the graph
        
        # build visit virtual nodes
        nonpad = adm_index_p != self.adm_pad_id
        adm_index_used_p = adm_index_p[nonpad] # adm_index非pad部分
        adm_ids_unique, adm_lid_nonpad = torch.unique(adm_index_used_p, return_inverse=True)
        num_visit_p = adm_ids_unique.numel()  # int, number of visits for patient
        adm_lid_full = torch.full_like(token_types_p, fill_value=-1) # [L]
        adm_lid_full[nonpad] = adm_lid_nonpad
        hg['visit'].x = visit_embed_p[:num_visit_p, :]
        hg['visit'].num_nodes = num_visit_p
        
        # build medical code nodes
        gid_occ_embed = seq_embed_p[occ_pos, :]
        hg['occ'].x = gid_occ_embed
        hg['occ'].num_nodes = num_occ

        # build edges between occ nodes and virtual visit nodes
        occ_adm_lid = adm_lid_full[occ_pos]
        assert (occ_adm_lid != -1).all(), "occ_adm_lid contains -1"
        e_v2o = torch.stack([occ_adm_lid, torch.arange(num_occ, device=self.device)], dim=0)
        e_o2v = torch.stack([torch.arange(num_occ, device=self.device), occ_adm_lid], dim=0)
        hg['visit','contains','occ'].edge_index = e_v2o
        hg['occ','contained_by','visit'].edge_index = e_o2v
        
        # build forward edges between virtual visit nodes
        if num_visit_p > 1:
            src = torch.arange(0, num_visit_p - 1, device=self.device)
            dst = torch.arange(1, num_visit_p, device=self.device)
            e_next = torch.stack([src, dst], dim=0) # [2, num_visit_p-1]
        else:
            e_next = torch.empty(2, 0, dtype=torch.long, device=self.device)
        hg['visit','next','visit'].edge_index = e_next
        return hg

    def process_gnn_out(self, gnn_out, num_visits, V):
        """Process the output of the GNN layer.

        Args:
            gnn_out (Tensor): The output of the GNN layer. Shape [sum(num_visits), d]
            num_visits (Tensor): A tensor containing the number of visits for each patient.
            V (int): The maximum number of visits.

        Returns:
            Tensor: The processed visit embeddings. Shape [B, V, d]
        """
        B = len(num_visits)
        # 计算每个批次的累积偏移量
        cumsum = torch.cumsum(num_visits, dim=0)  # [B]
        offsets = torch.cat([torch.tensor([0], device=self.device), cumsum[:-1]])  # [B]

        # 创建索引以从 gnn_out 中提取所有批次的嵌入
        indices = torch.arange(sum(num_visits), device=self.device)  # [N]
        batch_indices = torch.repeat_interleave(torch.arange(B, device=self.device), num_visits)  # [N]
        visit_pos = indices - offsets[batch_indices]  # [N]，每个嵌入的相对位置

        # 创建目标张量 visit_emb_pad，初始化为零
        visit_emb_pad = torch.zeros(B, V, self.d_model, device=self.device, dtype=gnn_out.dtype)  # [B, V, d]

        # 创建掩码，选择有效位置 (visit_pos < V 且 visit_pos < num_visits)
        mask = (visit_pos < V) & (visit_pos < num_visits[batch_indices])  # [N]
        valid_indices = indices[mask]  # [N_valid]
        valid_batch_indices = batch_indices[mask]  # [N_valid]
        valid_visit_pos = visit_pos[mask]  # [N_valid]

        # 使用 scatter 将 gnn_out 的值分配到 visit_emb_pad
        visit_emb_pad[valid_batch_indices, valid_visit_pos] = gnn_out[valid_indices]
        return visit_emb_pad
    
    def prepare_tf_input(self, task_id_embed, seq_embed, visit_embed, layer_i, 
                         input_ids, adm_index, token_types, num_visits, diag_code_group_dicts):
        """Prepare the input for the Transformer layer.
        Args:
            task_id_emb (Tensor): Task ID embeddings. Shape [B, 1, d]
            seq_embed (Tensor): Sequence embeddings. Shape [B, L, d]
            visit_embed (Tensor): Visit embeddings. Shape [B, V, d]
            layer_i (int): The current layer index.
            adm_index (tensor): The admission index. Shape [B, L]
            token_types (Tensor): Token types. Shape [B, L]
            num_visits (Tensor): Number of visits for each patient. Shape [B]
            diag_code_group_dicts (list): B dictionaries mapping group code idx to their corresponding diag code idx.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Processed inputs for the Transformer layer.
        """
        
        B, L, d = seq_embed.shape
        V = visit_embed.shape[1]

        # Part 1: prepare main seq embedding x
        # important: initiate new tensor to ensure safe autograd
        x = torch.empty(B, 1 + L + V, d, device=seq_embed.device, dtype=seq_embed.dtype)
        x[:, 0:1, :] = task_id_embed
        x[:, 1:1 + L, :] = seq_embed
        x[:, 1 + L:, :] = visit_embed
        
        # we already have token_types for main seq, just prepare token types for visit nodes
        # here it is out of the if branch because it is needed in the mask making
        arange_V = torch.arange(1, V + 1, device=self.device, dtype=torch.long)[None, :]  # [V]: [1, 2, 3, ..., V]
        n_v = num_visits.view(B, 1)  # [B, 1]
        visit_adm_index = torch.where(arange_V <= n_v, arange_V, torch.full((B, V), self.adm_pad_id, device=self.device, dtype=torch.long))  # [B, V]
        visit_type_id = torch.full((B, V), self.node_type_id_dict['visit'], dtype=torch.long, device=self.device)  # [B, V]
        visit_type_id_mask = (visit_adm_index != self.adm_pad_id).long() # [B, V]
        visit_type_id = visit_type_id * visit_type_id_mask # [B, V]
        token_types = torch.cat([token_types, visit_type_id], dim=1)  # [B, L+V]

        # if it is the first time transformer going through, we need to add extra information of admission index and token types
        if (layer_i == 0) or (layer_i == 1 and self.layer_types[0] == 'gnn'):
            adm_index = torch.cat([adm_index, visit_adm_index], dim=1)  # [B, L+V]
            # transform into embedding and add
            adm_index_embed = self.adm_index_emb(adm_index)
            token_type_embed = self.type_emb(token_types)
            x_non_task = x[:, 1:, :]
            x_non_task.add_(adm_index_embed).add_(token_type_embed)
            x[:, 1:, :] = x_non_task
        else:
            x = x
            
        # part 2: prepare mask (src_key_padding_mask and attn_mask)
        task_pad_mask = torch.zeros((B, 1), dtype=torch.bool, device=self.device) # [B, 1]
        seq_pad_mask = (input_ids == self.seq_pad_id) # [B, L], bool
        visit_pad_mask = (visit_adm_index == self.adm_pad_id) # [B, V], bool
        src_key_padding_mask = torch.cat([task_pad_mask, seq_pad_mask, visit_pad_mask], dim=1)  # [B, 1+L+V]
        token_types_w_task = torch.cat([torch.full((B, 1), -1, device=self.device), token_types], dim=1)
        diag_code_group_dicts_w_task = [{k + 1: [v_i + 1 for v_i in v] for k, v in d.items()} for d in diag_code_group_dicts]
        attn_mask = self.build_attn_mask(token_types_w_task, 
                                         forbid_map=self.forbid_map, 
                                         num_heads=self.num_attn_heads,
                                         allow_attn_dicts=diag_code_group_dicts_w_task)
        assert attn_mask.dtype == src_key_padding_mask.dtype, f"attn_mask dtype ({attn_mask.dtype}) and src_key_padding_mask dtype ({src_key_padding_mask.dtype}) must match"
        return x, src_key_padding_mask, attn_mask
    
    def process_tf_out(self, h, L, V):
        return h[:, 0:1, :], h[:, 1:1 + L, :], h[:, 1 + L:, :]  # [B, 1, d], [B, L, d], [B, V, d]
        
    @staticmethod
    def build_attn_mask(token_types, forbid_map, num_heads, allow_attn_dicts):
        B, L = token_types.shape
        device = token_types.device
        
        if forbid_map == None:
            mask = torch.zeros((B, L, L), dtype=torch.bool, device=device)
        else:
            # 收集所有出现的 token 类型
            observed = torch.unique(token_types)
            for q_t, ks in forbid_map.items():
                observed = torch.unique(torch.cat([observed, torch.tensor([q_t] + list(ks), device=device)]))
            type_list = observed.sort().values
            t2i = {t.item(): i for i, t in enumerate(type_list)}  # Map token types to indices
            T = len(type_list)

            # 构造禁止矩阵 (T, T)，单向关系
            ban_table = torch.zeros((T, T), dtype=torch.bool, device=device)
            for q_t, ks in forbid_map.items():
                if q_t in t2i:
                    qi = t2i[q_t]
                    for k_t in ks:
                        if k_t in t2i:
                            ban_table[qi, t2i[k_t]] = True  # 只设置 q -> k 的禁止

            # 向量化映射 token_types 到类型索引
            mapping = torch.zeros_like(type_list, dtype=torch.long, device=device)
            for t, i in t2i.items():
                mapping[type_list == t] = i
            q_idx = mapping[torch.searchsorted(type_list, token_types.unsqueeze(-1))]
            k_idx = mapping[torch.searchsorted(type_list, token_types.unsqueeze(-2))]

            # 查询 ban_table 得到 (B, L, L)
            mask = ban_table[q_idx, k_idx].to(torch.bool)
        
        # —— 追加 allow 约束：白名单式 —— 
        if allow_attn_dicts is not None:
            for b, dct in enumerate(allow_attn_dicts):
                for q, ks in dct.items():
                    q_idx = int(q)
                    k_idx = torch.as_tensor(ks, device=device, dtype=torch.long)
                    if k_idx.numel() == 0:
                        continue
                    mask[b, q_idx, k_idx] = False
        
        # 扩展到 num_heads
        mask = mask.unsqueeze(1).expand(B, num_heads, L, L)
        mask = mask.reshape(B * num_heads, L, L)
        return mask