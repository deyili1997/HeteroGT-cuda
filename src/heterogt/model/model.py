import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch as HeteroBatch
from heterogt.model.layer import DiseaseOccHetGNN, BinaryPredictionHead, MultiPredictionHead, HierTransformerLayer, CLSQueryMHA

class HeteroGT(nn.Module):
    def __init__(self, tokenizer, d_model, num_heads, layer_types, max_num_adms, device, task, label_vocab_size, 
                 attn_mask_dicts, use_cls_cat):
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
        self.node_type_id_dict = {'[PAD]': 0, '[CLS]': 1, 'diag': 2, 'med': 3, 'lab': 4, 'pro': 5, 'group': 6, 'visit': 7}
        self.graph_node_types = ['diag']
        self.attn_mask_dicts = attn_mask_dicts
        assert len(attn_mask_dicts) == sum(lt == 'tf' for lt in self.layer_types), "attn_masks length does not match number of transformer layers"
        self.use_cls_cat = use_cls_cat

        # embedding layers
        self.token_emb = nn.Embedding(self.global_vocab_size, d_model, padding_idx=self.seq_pad_id) # special tokens (pad and cls), age, group_code, diag, med, lab, pro
        self.type_emb = nn.Embedding(self.n_type + 4, d_model, padding_idx=self.type_pad_id) # + 4 for [PAD], [CLS], group code and visit code
        self.adm_index_emb = nn.Embedding(self.max_num_adms + 3, d_model, padding_idx=self.adm_pad_id) # +3 for pad, cls and group. Group is using max_num_adms + 1 as adm index. [CLS] is using max_num_adms + 2 as adm index
        
        # stack together
        self.stack_layers = nn.ModuleList(self.make_gnn_layer() if layer_type == 'gnn' else self.make_tf_layer()
            for layer_type in self.layer_types
        )
        
        
        # prediction head
        if self.use_cls_cat:
            # CLS query layer
            self.cls_MHA = CLSQueryMHA(d_model=d_model, num_heads=num_heads, 
                                    attn_token_type=[self.node_type_id_dict['[CLS]'], self.node_type_id_dict['group'], self.node_type_id_dict['visit']], 
                                    use_raw_value_agg=True, fallback_to_cls=True)
            out_dim = 2 * self.d_model
        else:
            out_dim = self.d_model
            
        if task in ["death", "stay", "readmission"]:
            self.cls_head = BinaryPredictionHead(out_dim)
        else:
            self.cls_head = MultiPredictionHead(out_dim, label_vocab_size)

    def make_tf_layer(self):
        assert self.d_model % self.num_attn_heads == 0, "Invalid model and attention head dimensions"
        tf_layer = HierTransformerLayer(d_model=self.d_model, num_heads=self.num_attn_heads)
        return tf_layer

    def make_gnn_layer(self):
        return DiseaseOccHetGNN(d_model=self.d_model, heads=self.num_attn_heads)
    
    def forward(self, input_ids, token_types, adm_index, age_ids, diag_code_group_dicts):
        """Forward pass for the model.

        Args:
            input_ids (Tensor): Input token IDs. Shape of [B, L], including tokens of [CLS], diag, med, lab, pro, and group code, no visit code yet
            token_types (Tensor): Token type IDs. Shape of [B, L], including tokens of [CLS], diag, med, lab, pro, and group code, no visit code yet
            adm_index (Tensor): Admission index IDs. Shape of [B, L], including tokens of [CLS], diag, med, lab, pro, and group code, no visit code yet
            age_ids (Tensor): Age IDs. Shape of [B, V]
            diag_code_group_dicts (list): len of B, dictionaries mapping group code idx to their corresponding diag code idx.

        Returns:
            Tensor: Output logits. Shape of [B, label_size]
        """
        
        B, L = input_ids.shape
        assert len(diag_code_group_dicts) == B, "diag_code_group_dicts length does not match batch size"
        V = age_ids.shape[1]
        num_visits = (age_ids != self.age_pad_id).sum(dim=1) # [B], number of visits for each patient in the batch
        
        # 基础表示
        seq_embed = self.token_emb(input_ids)  # [B, L, d]
        visit_embed = self.token_emb(age_ids) # [B, V, d]

        # src_key_padding_mask, token_types, adm_index 不依赖于transformer的层数，提前准备
        src_key_padding_mask = torch.cat([(input_ids == self.seq_pad_id), (age_ids == self.age_pad_id)], dim=1)
        assert src_key_padding_mask.shape == (B, L + V), "src_key_padding_mask shape mismatch"
        # we already have token_types for main seq, just prepare token types for visit nodes
        arange_V = torch.arange(1, V + 1, device=self.device, dtype=torch.long)[None, :]  # [V]: [1, 2, 3, ..., V]
        n_v = num_visits.view(B, 1)  # [B, 1]
        visit_adm_index = torch.where(arange_V <= n_v, arange_V, torch.full((B, V), self.adm_pad_id, device=self.device, dtype=torch.long))  # [B, V]
        visit_type_id = torch.full((B, V), self.node_type_id_dict['visit'], dtype=torch.long, device=self.device)  # [B, V]
        visit_type_id_mask = (visit_adm_index != self.adm_pad_id).long() # [B, V]
        visit_type_id = visit_type_id * visit_type_id_mask # [B, V]
        token_types_full = torch.cat([token_types, visit_type_id], dim=1)  # [B, L+V]
        adm_index_full = torch.cat([adm_index, visit_adm_index], dim=1)  # [B, L+V]

        # run through layers
        tf_layers_count = 0
        for i, layer_type in enumerate(self.layer_types):
            if layer_type == 'gnn': # the only purpose is just to update visit_embed
                seq_embed_det = seq_embed.detach()
                visit_embed_det = visit_embed.detach()
                hg_batch = self.build_graph_batch(seq_embed_det, token_types, self.graph_node_types, visit_embed_det, adm_index) # num_visits is a 1d tensor of [B]
                gnn_out = self.stack_layers[i](hg_batch)['visit']  # extract virtual visit node representations
                visit_embed = self.process_gnn_out(gnn_out, num_visits, V) # [B, V, d]
                assert visit_embed.shape == (B, V, self.d_model), "GNN output shape mismatch"
            elif layer_type == 'tf':
                x, attn_mask_l = self.prepare_tf_input(seq_embed, visit_embed, tf_layers_count, adm_index_full, token_types_full, self.attn_mask_dicts, diag_code_group_dicts)
                h = self.stack_layers[i](x, src_key_padding_mask, attn_mask_l) # [B, 1+L+V, d]
                seq_embed, visit_embed = self.process_tf_out(h, L, V) # # [B, 1, d], [B, L, d], [B, V, d]
                tf_layers_count += 1
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
        if self.use_cls_cat:
            cls = h[:, 0, :]
            agg = self.cls_MHA(h, token_types_full)
            out = torch.cat([cls, agg], dim=1)
        else:
            out = h[:, 0, :]
        logits = self.cls_head(out)
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
        B = seq_embed.shape[0]

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
        # trim off cls and group code.
        mask = (token_types_p != self.node_type_id_dict['group']) & (token_types_p != self.node_type_id_dict['[CLS]'])
        seq_embed_p = seq_embed_p[mask, :]
        adm_index_p = adm_index_p[mask]
        token_types_p = token_types_p[mask]
        
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
    
    def prepare_tf_input(self, seq_embed, visit_embed, layer_i, adm_index, token_types, attn_mask_dicts, diag_code_group_dicts):
        """Prepare the input for the Transformer layer.
        Args:
            seq_embed (Tensor): Sequence embeddings. Shape [B, L, d]
            visit_embed (Tensor): Visit embeddings. Shape [B, V, d]
            layer_i (int): The current layer index.
            adm_index (tensor): The admission index. Shape [B, L]
            token_types (Tensor): Token types. Shape [B, L]
            attn_mask_dicts (list): len of number of transformer layers, general masking rules for transformer layers
            diag_code_group_dicts (list): B dictionaries mapping group code idx to their corresponding diag code idx.

        Returns:
            Tuple[Tensor, Tensor]: Processed inputs for the Transformer layer.
        """
        B, L, d = seq_embed.shape
        V = visit_embed.shape[1]

        # Part 1: prepare main seq embedding x
        # important: initiate new tensor to ensure safe autograd
        x = torch.empty(B, L + V, d, device=seq_embed.device, dtype=seq_embed.dtype) # [B, L+V, d]
        x[:, :L, :] = seq_embed
        x[:, L:, :] = visit_embed

        # if it is the first time transformer going through, we need to add extra information of admission index and token types
        if (layer_i == 0) or (layer_i == 1 and self.layer_types[0] == 'gnn'):
            # transform into embedding and add
            adm_index_embed = self.adm_index_emb(adm_index) # [B, L+V, d]
            token_type_embed = self.type_emb(token_types) # [B, L+V, d]
            x.add_(adm_index_embed).add_(token_type_embed)
            assert x.shape == (B, L + V, self.d_model), "Input X shape mismatch"
        else: # if not first layer of transformer then do nothing
            x = x
            
        # part 2: prepare attn mask
        # sanity check before building attn mask B
        for i, d in enumerate(diag_code_group_dicts):
            if not d:
                continue
            keys = torch.as_tensor(list(d.keys()), device=token_types.device)
            vals = torch.as_tensor([v for vs in d.values() for v in vs], device=token_types.device)

            assert (token_types[i, keys] == self.node_type_id_dict['group']).all(), f"keys check fail at {i}"
            assert (token_types[i, vals] == self.node_type_id_dict['diag']).all(), f"values check fail at {i}"

        # build attn mask
        attn_mask = self.build_attn_mask(token_types,
                                         forbid_map=attn_mask_dicts[layer_i],
                                         num_heads=self.num_attn_heads,
                                         allow_attn_dicts=diag_code_group_dicts)

        return x, attn_mask

    def process_tf_out(self, h, L, V):
        # h: [B, L+V, d]。其中 h[:, 0, :] 为 [CLS]（或序列首位）
        assert h.shape[1] == L + V, "Transformer output length mismatch"
        return h[:, :L, :], h[:, L:, :]
        
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
                if dct is None:
                    continue
                for q, ks in dct.items():
                    q_idx = int(q)
                    k_idx = torch.as_tensor(ks, device=device, dtype=torch.long)
                    if k_idx.numel() == 0:
                        continue
                    mask[b, q_idx, k_idx] = False
        
        # make sure each token can at least attend to itself
        diag_mask = torch.eye(L, dtype=torch.bool, device=mask.device).unsqueeze(0).expand(B, L, L)
        mask = mask & ~diag_mask

        # 扩展到 num_heads
        mask = mask.unsqueeze(1).expand(B, num_heads, L, L)
        mask = mask.reshape(B * num_heads, L, L)
        return mask