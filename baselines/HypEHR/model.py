import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import Embedding, BinaryPredictionHead, MaskedPredictionHead, HalfNLHconv

class SetGNN(nn.Module):
    def __init__(self, config, tokenizer, norm=None):
        super(SetGNN, self).__init__()

        self.global_vocab_size = config["global_vocab_size"]
        self.tokenizer = tokenizer
        self.use_type_embed = config["hg_use_type_embed"]
        if self.use_type_embed:
            self.token_type_list = self._build_token_type_list()
        self.all_num_layers = config["hg_all_num_layers"]
        self.dropout = config["hg_dropout"]
        self.aggr = config["hg_aggregate"]
        self.token_embedding = Embedding(vocab_size=self.global_vocab_size,
                                         hidden_size=config["hg_hidden_size"],
                                         pad_id=self.tokenizer.convert_tokens_to_ids(["[PAD]"], voc_type="all")[0],
                                         dropout_prob=0.0)
        self.type_embedding = Embedding(vocab_size=len(config["predicted_token_type"]) + 2, 
                                        hidden_size=config["hg_hidden_size"], 
                                        pad_id=0, 
                                        dropout_prob=0.0)
        self.NormLayer = config["hg_normalization"]
        self.InputNorm = True
        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()

        self.V2EConvs.append(HalfNLHconv(in_dim=config["hg_hidden_size"], hid_dim=config["hg_hidden_size"], out_dim=config["hg_hidden_size"], num_layers=config["MLP_num_layers"],
                                         dropout=self.dropout, Normalization=self.NormLayer, InputNorm=self.InputNorm, heads=config["hg_num_heads"],
                                         attention=config["PMA"]))
        self.bnV2Es.append(nn.BatchNorm1d(config["hg_hidden_size"]))
        for i in range(self.all_num_layers):
            self.E2VConvs.append(HalfNLHconv(in_dim=config["hg_hidden_size"], hid_dim=config["hg_hidden_size"], out_dim=config["hg_hidden_size"],
                                             num_layers=config["MLP_num_layers"], dropout=self.dropout, Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm, heads=config["hg_num_heads"], attention=config["PMA"]))
            self.bnE2Vs.append(nn.BatchNorm1d(config["hg_hidden_size"]))
            self.V2EConvs.append(HalfNLHconv(in_dim=config["hg_hidden_size"], hid_dim=config["hg_hidden_size"], out_dim=config["hg_hidden_size"],
                                             num_layers=config["MLP_num_layers"], dropout=self.dropout, Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm, heads=config["hg_num_heads"], attention=config["PMA"]))
            if i < self.all_num_layers-1:  # No need to add BN for the last layer
                self.bnV2Es.append(nn.BatchNorm1d(config["hg_hidden_size"]))
        

        if config["task"] in ["death", "stay", "readmission"]:
            self.downstream_cls = BinaryPredictionHead(config["hg_hidden_size"] * (self.all_num_layers + 1))
        else:
            self.downstream_cls = MaskedPredictionHead(config["hg_hidden_size"] * (self.all_num_layers + 1), config["label_vocab_size"])
        
        self.final_layer = nn.Linear(config["hg_hidden_size"] * (self.all_num_layers + 1), config["hg_hidden_size"])
        
    def load_weight(self, checkpoint_dict, strict=False):
        missing_keys, unexpected_keys = self.load_state_dict(checkpoint_dict, strict=strict)
        # if missing_keys:
        #     print(f"[Warning] Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"[Warning] Unexpected keys: {unexpected_keys}")

    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()
        
    def _build_token_type_list(self):
        vocab_size = self.global_vocab_size
        token_type_list = [0] * vocab_size  # default: PAD=0

        # Special tokens
        special_range = self.tokenizer.token_id_range("special")
        for i in range(special_range[0], special_range[1]):
            token_type_list[i] = i  # CLS or other special tokens

        # Define mapping: voc_type → type_id
        type_mapping = {"diag": 2, "med": 3, "lab": 4, "pro": 5}

        for voc_type, type_id in type_mapping.items():
            start, end = self.tokenizer.token_id_range(voc_type)
            for i in range(start, end):
                token_type_list[i] = type_id

        # ====== ASSERTS ======
        from collections import Counter
        counter = Counter(token_type_list)

        num_special_tokens = self.tokenizer.token_id_range("special")[1] - self.tokenizer.token_id_range("special")[0]

        assert counter[0] == 1, f"PAD (type 0) count should be 1, got {counter[0]}"
        assert counter[1] == num_special_tokens - 1, f"CLS (type 1) count should be 1 if only 2 special tokens, got {counter[1]}"

        for voc_type, type_id in type_mapping.items():
            expected = self.tokenizer.token_number(voc_type)
            actual = counter[type_id]
            assert actual == expected, f"{voc_type} (type {type_id}) count mismatch: expected {expected}, got {actual}"

        return token_type_list  # length = vocab size, int ∈ {0,1,2,3,4,5}

    def forward(self, data, global_node_ids, last_visit_indices, exp_flags, edge_weight=None):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
        #             The data should contain the follows
        #             data.x: node features

        # edge_index is V (0-99) -> E (100-2199)
        # x is of shape [100 (nodes), 64 (featrue dimension)]
        # norm is all identical to 1.0
        # we need to put edge_index and norm to device
        word_embed, edge_index, norm = self.token_embedding(global_node_ids), data.edge_index.to(global_node_ids.device), data.norm.to(global_node_ids.device) 
        # print("input x.shape:", x.shape)
        
        if self.use_type_embed:
            token_type_tensor = torch.tensor(self.token_type_list, device=global_node_ids.device)
            type_ids = token_type_tensor[global_node_ids]  # [B_num_nodes]
            type_embed = self.type_embedding(type_ids)     # [B_num_nodes, D]
        else:
            type_embed = torch.zeros_like(word_embed)
        x = word_embed + type_embed
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory (actually wont matter much)
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0) # flip the edge_index, E -> V

        vec = [] # collect for jumping knowledge
        x = F.dropout(x, p=0.2, training=self.training)  # Input dropout

        scale = 1
        eps = 1e-5
        for i, _ in enumerate(self.E2VConvs):
            x, weight_tuple = self.V2EConvs[i](x, edge_index, norm, self.aggr, edge_weight=edge_weight)
            # PairNorm
            x = x - x.mean(dim=0, keepdim=True)
            x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
            # Jumping Knowledge
            vec.append(x)
            x = self.bnV2Es[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # print("x.shape after V2EConv:", x.shape)

            x, weight_tuple = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr, edge_weight=edge_weight)
            # PairNorm
            x = x - x.mean(dim=0, keepdim=True)
            x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
            node_feat = x
            x = self.bnE2Vs[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # print("x.shape after E2VConv:", x.shape)

        x, weight_tuple = self.V2EConvs[-1](x, edge_index, norm, self.aggr, edge_weight=edge_weight)
        # PairNorm
        x = x - x.mean(dim=0, keepdim=True)
        x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
        edge_feat = x
        # Jumping Knowledge
        vec.append(x)

        x = torch.cat(vec, dim=1)
        last_visit_embeds = x[last_visit_indices] 
        prediction = self.downstream_cls(last_visit_embeds[exp_flags])
        return prediction