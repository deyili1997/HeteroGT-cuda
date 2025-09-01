import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_scatter import scatter_add
from collections import Counter
import numpy as np


def _id2multi_hot(ids, dim):
    # return a multi-hot vector of size dim, where ids are the indices to be set to 1
    multi_hot = torch.zeros(dim).long()
    multi_hot[ids] = 1
    return multi_hot

class FineTuneEHRDataset(Dataset):
    def __init__(self, ehr_finetune_data, tokenizer, token_type, task):
        self.tokenizer = tokenizer
        self.task = task

        def transform_data(data, task):
            hadm_records, exp_flags = {}, {}
            genders, ages = {}, {}
            labels = {}
            for subject_id in data['SUBJECT_ID'].unique():
                item_df = data[data['SUBJECT_ID'] == subject_id]
                exp_flag = item_df['EXP_FLAG'].values[0]

                patient, age = [], []
                for _, row in item_df.iterrows():
                    admission = []
                    hadm_id = row['HADM_ID']
                    if "diag" in token_type:
                        admission.append(list(row['ICD9_CODE']))
                    if "med" in token_type:
                        admission.append(list(row['NDC']))
                    if "lab" in token_type:
                        admission.append(list(row['LAB_TEST'])) # admission of structure[[diag], [med], [pro], [lab]]
                    if "pro" in token_type:
                        admission.append(list(row['PRO_CODE']))
                    patient.append(admission)
                    age.append(row['AGE'])
                    
                    if exp_flag == False:
                        hadm_records[hadm_id] = list(patient)
                        genders[hadm_id] = row["GENDER"]
                        ages[hadm_id] = list(age)
                        labels[hadm_id] = None
                        exp_flags[hadm_id] = exp_flag
                    else:
                        if task in ["death", "stay", "readmission"]:  # binary prediction
                            hadm_records[hadm_id] = list(patient)
                            genders[hadm_id] = row["GENDER"]
                            ages[hadm_id] = list(age)
                            labels[hadm_id] = [row["DEATH"], row["STAY_DAYS"], row["READMISSION"]]
                            exp_flags[hadm_id] = exp_flag
                        else: # next diagnosis prediction
                            flag = row["NEXT_DIAG_6M"] if task == "next_diag_6m" else row["NEXT_DIAG_12M"]
                            if str(flag) != "nan":  # only include the admission with next diagnosis
                                # in other words, only admissions with future diagnosis labels are retained.
                                hadm_records[hadm_id] = list(patient)
                                genders[hadm_id] = row["GENDER"]
                                ages[hadm_id] = list(age)
                                label = row["NEXT_DIAG_6M_PHENO"] if task == "next_diag_6m" else row["NEXT_DIAG_12M_PHENO"]
                                labels[hadm_id] = list(label)
                                exp_flags[hadm_id] = exp_flag
            return hadm_records, genders, ages, labels, exp_flags
        self.records, self.genders, self.ages, self.labels, self.exp_flags = transform_data(ehr_finetune_data, task)

    def __len__(self):
        return len(self.records)
    
    def get_ids(self):
        return list(self.records.keys())
    
    def __getitem__(self, idx):
        hadm_id = list(self.records.keys())[idx]
        exp_flag = self.exp_flags[hadm_id]
        input_tokens = ["[CLS]"]
        token_types = [1] # 0 is used for PAD token and 1 is used for CLS token
        adm_index = [1] # 0 is used for PAD token and 1 is used for CLS token
        gender = self.genders[hadm_id]
        ages = self.ages[hadm_id]
        age_genders = [ages[-1] + "_" + gender] # CLS corresponds to the last visit's age and gender  
    
        # iterate through all encounters in this encounter sequence
        for idx, adm in enumerate(self.records[hadm_id]):
            adm_tokens = []
            adm_token_types = []
            # iterate through all types of tokens in this encounter
            for i in range(len(adm)):
                cur_tokens = list(adm[i])
                adm_tokens.extend(cur_tokens)
                # here + 2 is used to distinguish the PAD and CLS token
                adm_token_types.extend([i + 2] * len(cur_tokens)) 
                
            input_tokens.extend(adm_tokens)
            token_types.extend(adm_token_types)
            # 0 is used for PAD token and 1 is used for CLS token, here + 2 is used to distinguish the PAD and CLS token
            adm_index.extend([idx + 2] * len(adm_tokens)) 
            age_genders.extend([ages[idx] + "_" + gender] * len(adm_tokens))
        
        if exp_flag == True:
        # build labels based on the task
            if self.task == "death":
                # predict if the patient will die in the hospital
                labels = torch.tensor([self.labels[hadm_id][0]]).float()
            elif self.task == "stay":
                # predict if the patient will stay in the hospital for more than 7 days
                labels = (torch.tensor([self.labels[hadm_id][1]]) > 7).float()
            elif self.task == "readmission":
                # predict if the patient will be readmitted within 1 month
                labels = torch.tensor([self.labels[hadm_id][2]]).float()
            else:  # next diagnosis prediction
                # we abandon the last encounter insertion here (in the original paper) because we will use the CLS token as the patient representation
                # and this representation will further go through a hypergraph to do information exchange.
                label_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.labels[hadm_id], voc_type='diag'))
                labels = _id2multi_hot(label_ids, dim=self.tokenizer.token_number('diag'))
        else:
            labels = None

        # convert input_tokens to ids
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(input_tokens, voc_type = "all")], dtype=torch.long)
        # convert token_types to tensor
        token_types = torch.tensor([token_types], dtype=torch.long)
        # convert adm_index to tensor
        adm_index = torch.tensor([adm_index], dtype=torch.long)
        # convert age_gender_index to tensor
        age_gender_index = torch.tensor([self.tokenizer.convert_tokens_to_ids(age_genders, voc_type="age_gender")], dtype=torch.long)


        # sanity check
        assert input_ids.shape == token_types.shape == adm_index.shape == age_gender_index.shape, \
            f"Input IDs shape {input_ids.shape}, token types shape {token_types.shape}, adm index shape {adm_index.shape}, age gender index shape {age_gender_index.shape} do not match"

        return input_ids, token_types, adm_index, age_gender_index, labels, exp_flag

class FinetuneHGDataset(FineTuneEHRDataset):
    def __init__(self, ehr_finetune_data, tokenizer, token_type, task, level):
        # no masking
        super().__init__(ehr_finetune_data, tokenizer, token_type, task)
        self.level = level 
        # if on visit level, each hyperedge will be each patient's visit, and we will output the last visit index in a batch for prediction
        # if on patient level, each hyperedge will be the entire patient trajectory, and we will just simply output the row index in a batch for prediction

    def __getitem__(self, idx):
        hadm_id = list(self.records.keys())[idx]
        exp_flag = self.exp_flags[hadm_id]
        input_tokens = []
        for adm in self.records[hadm_id]:
            adm_tokens = []
            for i in range(len(adm)):
                cur_tokens = list(adm[i])
                adm_tokens.extend(list(set(cur_tokens)))

            if self.level == "visit":       
                input_tokens.append(self.tokenizer.convert_tokens_to_ids(adm_tokens, voc_type = "all"))
            else:  # patient level
                input_tokens.extend(self.tokenizer.convert_tokens_to_ids(adm_tokens, voc_type = "all"))

        if self.level == "patient":
            input_tokens = [list(dict.fromkeys(input_tokens))]  # preserves order while deduplicating, [[]] is for the unified patient representation

        if exp_flag == True:
            # build labels based on the task
            if self.task == "death":
                # predict if the patient will die in the hospital
                labels = torch.tensor([self.labels[hadm_id][0]]).float()
            elif self.task == "stay":
                # predict if the patient will stay in the hospital for more than 7 days
                labels = (torch.tensor([self.labels[hadm_id][1]]) > 7).float()
            elif self.task == "readmission":
                # predict if the patient will be readmitted within 1 month
                labels = torch.tensor([self.labels[hadm_id][2]]).float()
            else:  # next diagnosis prediction
                labels = torch.tensor(self.labels[hadm_id]).float()
        else:
            labels = None
        return input_tokens, labels, exp_flag


def ExtractV2E(data):
    # Assume edge_index = [V|E;E|V]
    edge_index = data.edge_index
#     First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

    num_nodes = data.n_x[0]
    if not ((data.n_x[0]+data.num_hyperedges[0]-1) == data.edge_index[0].max().item()):
        print('num_hyperedges does not match! 1')
        return
    cidx = torch.where(edge_index[0] == num_nodes)[0].min()  # cidx: [V...|cidx E...]
    data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)
    return data


def Add_Self_Loops(data):
    # update so we dont jump on some indices
    # Assume edge_index = [V;E]. If not, use ExtractV2E()
    edge_index = data.edge_index
    num_nodes = data.n_x[0]
    num_hyperedges = data.num_hyperedges[0]

    if not ((data.n_x[0] + data.num_hyperedges[0] - 1) == data.edge_index[1].max().item()):
        print('num_hyperedges does not match! 2')
        return

    hyperedge_appear_fre = Counter(edge_index[1].numpy())
    # store the nodes that already have self-loops
    skip_node_lst = []
    for edge in hyperedge_appear_fre:
        if hyperedge_appear_fre[edge] == 1:
            skip_node = edge_index[0][torch.where(
                edge_index[1] == edge)[0].item()]
            skip_node_lst.append(skip_node.item())

    new_edge_idx = edge_index[1].max() + 1
    new_edges = torch.zeros(
        (2, num_nodes - len(skip_node_lst)), dtype=edge_index.dtype)
    tmp_count = 0
    for i in range(num_nodes):
        if i not in skip_node_lst:
            new_edges[0][tmp_count] = i
            new_edges[1][tmp_count] = new_edge_idx
            new_edge_idx += 1
            tmp_count += 1

    data.totedges = num_hyperedges + num_nodes - len(skip_node_lst)
    edge_index = torch.cat((edge_index, new_edges), dim=1)
    # Sort along w.r.t. nodes
    _, sorted_idx = torch.sort(edge_index[0])
    data.edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    return data


def norm_contruction(data, option='all_one'):
    if option == 'all_one':
        data.norm = torch.ones_like(data.edge_index[0])

    elif option == 'deg_half_sym':
        edge_weight = torch.ones_like(data.edge_index[0])
        cidx = data.edge_index[1].min()
        Vdeg = scatter_add(edge_weight, data.edge_index[0], dim=0)
        HEdeg = scatter_add(edge_weight, data.edge_index[1]-cidx, dim=0)
        V_norm = Vdeg**(-1/2)
        E_norm = HEdeg**(-1/2)
        data.norm = V_norm[data.edge_index[0]] * \
            E_norm[data.edge_index[1]-cidx]

    return data


def batcher_SetGNN_finetune(device):
    def batcher_dev(batch):
        raw_input_tokens, labels, exp_flags = [feat[0] for feat in batch], [feat[1] for feat in batch], [feat[2] for feat in batch]
        
        # 预过滤
        labels_np = np.array(labels, dtype=object)
        mask_np = np.array(exp_flags, dtype=bool)
        filtered_labels = labels_np[mask_np]
        assert all(x is not None for x in filtered_labels)
        
        # 展平visits
        flat_visits = []
        last_visit_indices = []
        for i in range(len(raw_input_tokens)):
            start_idx = len(flat_visits)
            flat_visits.extend(raw_input_tokens[i])
            last_visit_indices.append(start_idx + len(raw_input_tokens[i]) - 1)
        
        if not flat_visits:
            # 空batch处理
            empty_data = Data(edge_index=torch.empty((2, 0), dtype=torch.long))
            empty_data.n_x = torch.tensor([0])
            empty_data.num_hyperedges = torch.tensor([0])
            return (empty_data, torch.empty(0, dtype=torch.long),
                   torch.tensor(last_visit_indices, dtype=torch.long),
                   torch.tensor(exp_flags, dtype=torch.bool),
                   torch.empty(0, dtype=torch.float))
        
        # 使用numpy加速节点收集
        try:
            # 尝试向量化操作
            all_visit_arrays = [np.array(visit) for visit in flat_visits if len(visit) > 0]
            if all_visit_arrays:
                all_nodes_array = np.concatenate(all_visit_arrays)
                global_node_ids = np.unique(all_nodes_array).tolist()
            else:
                global_node_ids = []
        except:
            # 回退到原始方法
            all_nodes = set()
            for visit in flat_visits:
                all_nodes.update(visit)
            global_node_ids = sorted(list(all_nodes))
        
        # 节点映射
        node_id_map = {nid: i for i, nid in enumerate(global_node_ids)}
        
        # 重映射和构建边
        edge_list = []
        num_nodes = len(global_node_ids)
        
        for h_id, visit in enumerate(flat_visits):
            hyperedge_id = num_nodes + h_id
            for node_id in visit:
                remapped_node = node_id_map[node_id]
                edge_list.extend([[remapped_node, hyperedge_id], 
                                [hyperedge_id, remapped_node]])
        
        # 转换为张量
        edge_index = torch.tensor(edge_list, dtype=torch.long).T.contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
        
        # 处理标签
        if len(filtered_labels[0]) > 1:
            labels = torch.tensor(np.array(filtered_labels.tolist(), dtype=np.float32)).float()
        else:
            labels = torch.cat([t.view(-1) for t in filtered_labels.tolist()]).float() if len(filtered_labels) > 0 else torch.empty(0, dtype=torch.float)
        
        # 构建数据对象
        data = Data(edge_index=edge_index)
        data.n_x = torch.tensor([num_nodes])
        data.num_hyperedges = torch.tensor([len(flat_visits)])
        
        data = ExtractV2E(data)
        data = Add_Self_Loops(data)
        data = norm_contruction(data)
        
        return (data, 
                torch.tensor(global_node_ids, dtype=torch.long), 
                torch.tensor(last_visit_indices, dtype=torch.long), 
                torch.tensor(exp_flags, dtype=torch.bool), 
                labels)
    
    return batcher_dev