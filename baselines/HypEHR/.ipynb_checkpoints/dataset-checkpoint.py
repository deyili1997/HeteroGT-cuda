import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_scatter import scatter_add
from collections import Counter


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
            hadm_records = {}
            genders, ages = {}, {}
            labels = {}
            for subject_id in data['SUBJECT_ID'].unique():
                item_df = data[data['SUBJECT_ID'] == subject_id]
                
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
                    
                    if task in ["death", "stay", "readmission"]:  # binary prediction
                        hadm_records[hadm_id] = list(patient)
                        genders[hadm_id] = row["GENDER"]
                        ages[hadm_id] = list(age)
                        labels[hadm_id] = [row["DEATH"], row["STAY_DAYS"], row["READMISSION"]]
                    else: # next diagnosis prediction
                        flag = row["NEXT_DIAG_6M"] if task == "next_diag_6m" else row["NEXT_DIAG_12M"]
                        if str(flag) != "nan":  # only include the admission with next diagnosis
                            # in other words, only admissions with future diagnosis labels are retained.
                            hadm_records[hadm_id] = list(patient)
                            genders[hadm_id] = row["GENDER"]
                            ages[hadm_id] = list(age)
                            label = row["NEXT_DIAG_6M_PHENO"] if task == "next_diag_6m" else row["NEXT_DIAG_12M_PHENO"]
                            labels[hadm_id] = list(label)
            return hadm_records, genders, ages, labels
        self.records, self.genders, self.ages, self.labels = transform_data(ehr_finetune_data, task)

    def __len__(self):
        return len(self.records)
    
    def get_ids(self):
        return list(self.records.keys())
    
    def __getitem__(self, idx):
        hadm_id = list(self.records.keys())[idx]
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

        return input_ids, token_types, adm_index, age_gender_index, labels

class FinetuneHGDataset(FineTuneEHRDataset):
    def __init__(self, ehr_finetune_data, tokenizer, token_type, task, level):
        # no masking
        super().__init__(ehr_finetune_data, tokenizer, token_type, task)
        self.level = level 
        # if on visit level, each hyperedge will be each patient's visit, and we will output the last visit index in a batch for prediction
        # if on patient level, each hyperedge will be the entire patient trajectory, and we will just simply output the row index in a batch for prediction

    def __getitem__(self, idx):
        hadm_id = list(self.records.keys())[idx]
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
            
        return input_tokens, labels


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

# batch_SetGNN should directly return a processed PyG data object, can be directly intake by SetGNN
def batcher_SetGNN_finetune(device):
    def batcher_dev(batch):
        raw_input_tokens, labels = [feat[0] for feat in batch], [feat[1] for feat in batch]
        flat_visits = []
        last_visit_indices = []
        
        for i in range(len(raw_input_tokens)):
            start_idx = len(flat_visits)
            flat_visits.extend(raw_input_tokens[i]) # here we flatten samples in a batch
            last_visit_indices.append(start_idx + len(raw_input_tokens[i]) - 1)
            
        # Collect all unique node ids in this batch
        all_nodes = set()
        for visit in flat_visits:
            all_nodes.update(visit)
        global_node_ids = sorted(list(all_nodes))  # ensure consistent order
        
        # build a mapping: global_id -> local_id
        node_id_map = {nid: i for i, nid in enumerate(global_node_ids)}
        
        # remap visits to local node ids
        def remap(visits):
            return [[node_id_map[nid] for nid in visit] for visit in visits]
        flat_visits = remap(flat_visits)
        
        # stack batch labels
        labels = torch.stack(labels, dim=0).float()
        
        # construct the Data object
        edge_list = []
        num_nodes = len(all_nodes)
        # convert flat_visits to a PyG Data object, we construct it as a bipartile graph to represent HG
        for h_id, visit in enumerate(flat_visits):
            for token in visit:
                edge_list.append([token, num_nodes + h_id])  # 节点id, 超边id
                edge_list.append([num_nodes + h_id, token])
                
        edge_index = torch.tensor(edge_list).T.contiguous()  # shape (2, num_edges)
        
        data = Data(edge_index=edge_index)
        data.n_x = torch.tensor([num_nodes])
        data.num_hyperedges = torch.tensor([len(flat_visits)])  # number of visits
        data = ExtractV2E(data)
        data = Add_Self_Loops(data)
        data = norm_contruction(data)
        return data, torch.tensor(global_node_ids, dtype=torch.long), torch.tensor(last_visit_indices, dtype=torch.long), labels
    return batcher_dev