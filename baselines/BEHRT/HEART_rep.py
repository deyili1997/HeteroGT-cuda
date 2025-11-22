import torch
import torch.nn as nn
import torch.nn.functional as F
from layers_rep import TransformerBlock
from gnn import DotAttnConv
from sklearn.metrics import precision_recall_fscore_support
from transformer_rel import EdgeTransformerBlock, EdgeModule

    
# Hierarchical Transformer
class HiTransformer(nn.Module):
    def __init__(self, config):
        super(HiTransformer, self).__init__()
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config["num_hidden_layers"])])
        # multi-layers transformer blocks, deep network
        if config["gat"] == "dotattn":
            self.cross_attentions = nn.ModuleList(
                [DotAttnConv(config["hidden_size"], config["hidden_size"], 
                             config["gnn_n_heads"], config["max_visit_size"], config["gnn_temp"]) \
                                 for _ in range(config["num_hidden_layers"])])
        elif config["gat"] == "None":
            self.cross_attentions = None


    def forward(self, x, edge_index, mask, visit_positions):
        # x of shape [B, L, D], where B is the batch size, L is the max visit code length, and D is the hidden size
        # edge_index of shape [2, E], where E is the number of edges in the graph
        # mask of shape [B, L, L], 1 is pad and 0 is valid token
        # visit_positions is a single vector
        # running over multiple transformer blocks
        for i in range(len(self.transformer_blocks)): # transformer + GNN interleaved layers
            x = self.transformer_blocks[i](x, mask)  # [B, L, D]
        return x #[B, L, D]


# multi-class classification task
class MaskedPredictionHead(nn.Module):
    def __init__(self, config, voc_size):
        super(MaskedPredictionHead, self).__init__()
        self.cls = nn.Sequential(
                nn.Linear(config["hidden_size"], config["hidden_size"]), 
                nn.ReLU(), 
                nn.Linear(config["hidden_size"], voc_size)
            )

    def forward(self, input):
        return self.cls(input)


# binary classification task
# predict whether each code is an anomaly or not
# reduce D -> 1
class BinaryPredictionHead(nn.Module):
    def __init__(self, config):
        super(BinaryPredictionHead, self).__init__()
        self.cls = nn.Sequential(
                nn.Linear(config["hidden_size"], config["hidden_size"]), 
                nn.ReLU(), 
                nn.Linear(config["hidden_size"], 1)
            )

    def forward(self, input):
        return self.cls(input)


class HBERTEmbeddings(nn.Module):
    def __init__(self, config):
        super(HBERTEmbeddings, self).__init__()
        # If specified, the entries at padding_idx do not contribute to the gradient; therefore, 
        # the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”. 
        self.word_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"], padding_idx=0)  # [PAD] token
        self.emb_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, input_ids):
        # embedding the indexed sequence to sequence of vectors
        words_embeddings = self.word_embeddings(input_ids)
        return self.emb_dropout(words_embeddings)


# Hierarchical Transformer with Edge Representation
class HiEdgeTransformer(nn.Module):
    def __init__(self, config):
        super(HiEdgeTransformer, self).__init__()

        self.edge_module = EdgeModule(config)
        
        self.transformer_blocks = nn.ModuleList(
            [EdgeTransformerBlock(config) for _ in range(config["num_hidden_layers"])])

        if config["gat"] == "dotattn":
            self.cross_attentions = nn.ModuleList(
                [DotAttnConv(config["hidden_size"], config["hidden_size"], 
                             config["gnn_n_heads"], config["max_visit_size"], 
                             config["gnn_temp"]) for _ in range(config["num_hidden_layers"])])
        elif config["gat"] == "None":
            self.cross_attentions = None

    def forward(self, x, x_types, edge_index, mask, visit_positions):
        edge_embs = self.edge_module(x, x_types)  # [B, L, L, D]
        for i in range(len(self.transformer_blocks)):
            # note that here 1 is pad and 0 is valid token
            x = self.transformer_blocks[i](x, edge_embs, mask)  # [B, L, D]
            if edge_index is not None and self.cross_attentions is not None:
                x = torch.cat([self.cross_attentions[i](x[:, 0], edge_index, visit_positions).unsqueeze(dim=1), 
                                x[:, 1:]], dim=1)  # communicate between visits
        return x


class HBERT_Pretrain(nn.Module):
    def __init__(self, config, tokenizer):
        super(HBERT_Pretrain, self).__init__()
        
        # we are using tree embeddings
        diag_range = tokenizer.token_id_range("diag")

        if config["diag_med_emb"] == "simple":
            self.embeddings = HBERTEmbeddings(config)
        
        # it is interesting to learn that the obj function is inside the model
        self.loss_fn = F.binary_cross_entropy_with_logits

        self.encoder = config["encoder"]
        if self.encoder == "hi":
            self.transformer = HiTransformer(config)
        elif self.encoder == "hi_edge":
            self.transformer = HiEdgeTransformer(config)
        else:
            raise NotImplementedError
        
        self.mask_token_id = config["mask_token_id"]  # {"diag":3, "med":4, "pro":5, "lab":6}
        predicted_token_type = config["predicted_token_type"]  # ["diag", "med", "pro", "lab"]
        label_vocab_size = config["label_vocab_size"]  # {token_type: vocab_size (type specific)}
        
        for token_type in predicted_token_type:
            # self.add_module(name, module)
            # A method from torch.nn.Module that registers a submodule under the given name.
            # Equivalent to self.name = module, but useful when the name is generated dynamically (i.e., as a string).
            self.add_module(f"{token_type}_cls", MaskedPredictionHead(config, label_vocab_size[token_type]))
            
    
    def forward(self, input_ids, token_types, edge_index, visit_positions, masked_labels):
        # one batch contains, B is batch size:
        # input_ids (all patient visit number in batch, max one visit code len), note that masked token not included. This is tokenized.
        # input_types (all patient visit number in batch, max one visit code len), note that masked token not included.
        # edge_index (2, sum of number of visit * number of visit of each patient in batch), for GNN of the batch graph connecting all patient visits.
        # visit_positions (all patient visit number in batch). Just a single vector [0, 1, 2, 0, 1, 2, 3, ...].
        # labels (is a list [diag (all patient visit number in batch, by all diag vocab num, i.e., multi-hot embedding), med, pro, lab]), 
        # anomaly_labels (all patient visit number in batch, max one visit code len (position of anomaly))
        
        pad_mask = (input_ids > 0) # bool matrix, filter pad (0) not mask (3,4,5,6)
        # pad_mask.unsqueeze(1) changes the shape to [B, 1, L]
        # repeat the pad_mask to match the shape of input_ids, i.e., [B, L, L]
        # pair_pad_mask is used to mask the padded positions in the pairwise attention
        # each arg in repeat corresponds to the multiplier to the dimensions of the tensor
        pair_pad_mask = pad_mask.unsqueeze(1).repeat(1, input_ids.size(1), 1)

        # embedding the indexed sequence to sequence of vectors
        # here we use B to denote the total number of patient visits in a batch
        # L is the max visit code length in the batch
        # D is the hidden size of the model
        x = self.embeddings(input_ids) # shape of [B, L, D]

        # ~pair_pad_mask
        # 1 means padded (mask out)
        # 0 means valid token (attend to it)
        if self.encoder == "hi":
            x = self.transformer(x, edge_index, ~pair_pad_mask, visit_positions) # [B, L, D]
        elif self.encoder == "hi_edge":
            x = self.transformer(x, token_types, edge_index, ~pair_pad_mask, visit_positions)

        ave_loss, loss_dict, perf_dict = 0, {}, {}
        for i, (token_type, mask_id) in enumerate(self.mask_token_id.items()):
            # take out [MASK_i] (one type of MASK token, representing the tokens that are truly masked), 
            # for pretraining task.
            masked_token_emb = x[input_ids == mask_id] # [B, 1, D]
            # learn the writing here
            prediction = self._modules[f"{token_type}_cls"](masked_token_emb)
            labels = masked_labels[i].to(input_ids.device)
            loss = self.loss_fn(prediction, labels)
            # loss = self.loss_fn(prediction.view(-1), masked_labels[i].view(-1).to(device), pos_weight=self.pos_weight.to(device))
            ave_loss += loss
            loss_dict[token_type] = loss.cpu().item()
            perf_dict[token_type] = self.evalute(prediction, labels)

        # loss dict have different code type loss as well as anomaly loss
        return ave_loss / len(loss_dict), loss_dict, perf_dict

    @staticmethod
    def evalute(prediction, labels):
        # Predictions
        probs = torch.sigmoid(prediction).detach().cpu().numpy()  # [N, C] — sigmoid for multi-label
        pred_classes = (probs >= 0.5).astype(int)                  # Threshold to get binary predictions

        # Ground truth
        true_labels = labels.detach().cpu().numpy()               # [N, C], should be float or int binary matrix

        # Precision, Recall, F1 — macro average across classes
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_classes, average='macro', zero_division=0
        )

        # Update performance dict
        perf_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        return perf_dict
    
    
class HBERT_Finetune(nn.Module):
    def __init__(self, config):
        super(HBERT_Finetune, self).__init__()

        if config["diag_med_emb"] == "simple":
            self.embeddings = HBERTEmbeddings(config)

        self.diag_mask_id = 3  # the idx of [MASK0] token
        self.task = config["task"]

        self.encoder = config["encoder"]
        if self.encoder == "hi":
            self.transformer = HiTransformer(config)
        elif self.encoder == "hi_edge":
            self.transformer = HiEdgeTransformer(config)


        if config["task"] in ["death", "stay", "readmission"]:
            self.downstream_cls = BinaryPredictionHead(config)
        else:
            self.downstream_cls = MaskedPredictionHead(config, config["label_vocab_size"])

    def load_weight(self, checkpoint_dict):
        # named_parameters: 获取当前模型的参数字典, 使用 dict() 将其转换为 key → parameter 的形式（key 是参数名字符串）
        # [('encoder.layer.0.weight', Parameter(...)), ('encoder.layer.0.bias', Parameter(...)), ...]
        # 遍历保存的 checkpoint_dict 中的每个参数 key
	    # 如果该 key 同时也存在于当前模型的参数 param_dict 中：
	    # 就把 checkpoint_dict 中的权重 复制 到当前模型对应参数的 .data 中
	    # 使用 .copy_() 是为了就地更新已有 tensor.data 的值，避免替换整个参数对象（否则会破坏计算图）
     
        # param.data.copy_() 是 原地修改 tensor 的值，绕过 autograd
        # .data 返回的是一个不受 autograd 管理的 Tensor 视图；
        # .copy_() 是一个就地操作（in-place），只改变数据本身，不创建新的张量；
        # 因此，它不会在 Autograd 中留下任何 操作节点（不会被记录在计算图中）；
        # 不会影响已有的梯度计算流程，也不会导致梯度断链或错误传播。

        param_dict = dict(self.named_parameters())
        for key in checkpoint_dict.keys():
            if key in param_dict:
                param_dict[key].data.copy_(checkpoint_dict[key])
    
    def forward(self, input_ids, token_types, edge_index, visit_positions, labeled_ids):
        # note that labeled_ids is the index of the last visit in each patient encounter
        # i.e., labeled_batch_idx in the batch.
        pad_mask = (input_ids > 0).unsqueeze(1).repeat(1, input_ids.size(1), 1) #[B, L, L]

        # embedding the indexed sequence to sequence of vectors
        x = self.embeddings(input_ids)

        if self.encoder == "hi":
            x = self.transformer(x, edge_index, ~pad_mask, visit_positions) # [B, L, D]
        elif self.encoder == "hi_edge":
            x = self.transformer(x, token_types, edge_index, ~pad_mask, visit_positions)


        if self.task in ["death", "stay", "readmission"]:
            # labeled_ids indicates each patient's last visit index
            # use CLS to make prediction
            prediction = self.downstream_cls(x[labeled_ids][:, 0])
        else:
            labeled_ids, labeled_x = input_ids[labeled_ids], x[labeled_ids]
            # get the position of manually added [MASK0] token
            # please note that this is the only masked token in the finetune dataset,
            # unlike pretrain dataset, where each code type has its own [MASK] token.
            masked_pos_embs = labeled_x[labeled_ids == self.diag_mask_id]
            prediction = self.downstream_cls(masked_pos_embs)
        return prediction