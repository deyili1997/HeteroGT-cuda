from torch.utils.data import Dataset
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
import random


def _pad_sequence(seqs, pad_id=0):
    # seqs: a list of tensor [n, m]
    max_len = max([x.shape[1] for x in seqs])
    # pad pad_id(0) on the right side 
    # return [N_visit, max_visit_len] of tensor here is the max length of the visit in a patient history
    return torch.cat([F.pad(x, (0, max_len - x.shape[1]), "constant", pad_id) for x in seqs], dim=0)

class HBERTPretrainEHRDataset(Dataset):
    def __init__(self, ehr_pretrain_data, tokenizer, token_type, 
                 mask_rate, anomaly_rate):
        self.tokenizer = tokenizer
        self.token_type = token_type
        self.mask_rate = mask_rate
        self.anomaly_rate = anomaly_rate
        self.records, self.ages, self.genders = self._transform_data(ehr_pretrain_data)
        # e.g. {0: 'diag', 1: 'med', 2: 'pro', 3: 'lab'}
        self.token_type_map = {i:t for i, t in enumerate(token_type)}

   
    def _transform_data(self, data):
        records, ages = {}, {}
        genders = {}
        for subject_id in data['SUBJECT_ID'].unique():
            # get one patient's hospitalization records
            item_df = data[data['SUBJECT_ID'] == subject_id]
            # get this patient's gender
            genders[subject_id] = [item_df.head(1)["GENDER"].values[0]]

            patient, age = [], []
            # iterate through each admission of the patient
            for _, row in item_df.iterrows():
                # please be note that until here the different medical codes are mixed together
                # use admission var to store the codes of one admission
                # admission = [[diag...], [med...], [pro], [lab]]
                admission = []
                
                if "diag" in self.token_type:
                    admission.append(list(row['ICD9_CODE']))
                if "med" in self.token_type:
                    admission.append(list(row['NDC']))
                if "pro" in self.token_type:
                    admission.append(list(row['PRO_CODE']))
                if "lab" in self.token_type:
                    admission.append(list(row['LAB_TEST']))
                patient.append(admission)
                # one patients may have multiple admissions, and each
                # admission can be associated with one age
                age.append(row['AGE']) # here row['AGE'] is a scalar, not a list
            records[subject_id] = list(patient)
            # records[subject_id] = [[[diag], [med], [pro], [lab]], [[diag], [med], [pro], [lab]]], 
            # note that the codes are in their original format, not tokenized
            ages[subject_id] = age
        return records, ages, genders # all dictionaries
    
    def _id2multi_hot(self, ids, dim):
        # return a multi-hot vector of size dim, where ids are the indices to be set to 1
        multi_hot = torch.zeros(dim)
        multi_hot[ids] = 1
        return multi_hot
    
    def __len__(self):
        return len(self.records)
    
    # get one patient's all admissions
    def __getitem__(self, item):
        subject_id = list(self.records.keys())[item]
        input_tokens, token_types, masked_labels, anomaly_labels = [], [], [None for _ in range(len(self.token_type))], []
        
        # loop through each admission of the patient
        for idx, adm in enumerate(self.records[subject_id]):
            # replace [CLS] token with age_gender token, please note that here the age_gender token has a type 0
            # evey admission will have an age_gender token representing [CLS], type 0, note that it means padding / CLS
            adm_tokens, adm_token_types, adm_masked_labels = \
                [str(self.ages[subject_id][idx]) + "_" + str(self.genders[subject_id][0])], [0], []  
            adm_anomaly_labels = []
            
            # loop through each type of code
            # one admission have many kinds of entities, [[diag], [med], ...]
            for i in range(len(adm)):  
                cur_tokens = list(adm[i]) # all codes belong to this type
                
                # randomly mask tokens for each type of code
                # note that in self.records, each admission is a list of lists, e.g. [[diag1, diag2], [med1, med2], ...]
                # without special tokens
                non_special_tokens_idx = [idx for idx, x in enumerate(cur_tokens)] # all indices of the current type tokens
                masked_tokens_idx = np.random.choice(non_special_tokens_idx, max(1, int(len(non_special_tokens_idx) * self.mask_rate)))
                masked_tokens = [cur_tokens[idx] for idx in masked_tokens_idx]
                masked_tokens_idx_ = set(masked_tokens_idx.tolist())  # for fast lookup
                non_masked_tokens = [cur_tokens[idx] for idx in non_special_tokens_idx if idx not in masked_tokens_idx_]
                
                # randomly replace tokens with other tokens for anomaly detection task
                if self.anomaly_rate > 0 and len(non_masked_tokens) > 0:
                    candidate_token_idx = [idx for idx, x in enumerate(non_masked_tokens)]
                    anomaly_tokens_idx = np.random.choice(candidate_token_idx, max(1, int(len(candidate_token_idx) * self.anomaly_rate)))
                    for ano_idx in anomaly_tokens_idx:
                        non_masked_tokens[ano_idx] = self.tokenizer.random_token(voc_type=self.token_type_map[i]) # switch to token of same code type
                        adm_anomaly_labels.append(len(adm_tokens) + ano_idx + 1)  # the position of the anomaly token, +1 for [MASK] tokens, [MASK] is between 2 code types
                adm_tokens.extend([f"[MASK{i}]"] + non_masked_tokens)  # [age_gender, MASK1, diag1, diag2, MASK2, med1, med2] of this one visit
                # these [MASK] tokens are for pretraining task of masked label prediction. Function just like [CLS].
                # please note that masked tokens are not included in adm_tokens, they are just for labels
                
                # [0, 1, 1, 2, 2] + 1 the first + 1 for the Age-gender token (type 0) and the second + 1 for [MASK] token
                adm_token_types.extend([i + 1] * (len(non_masked_tokens) + 1))  
                # [[diag1, diag2], [med1, med2]], since each type of code is masked separately, so the labels are also separated by type
                adm_masked_labels.append(masked_tokens) # this is not tokenized to ids yet.
            
            # here toenization is with the global vocab, not the type specific vocab
            input_tokens.append(torch.tensor([self.tokenizer.convert_tokens_to_ids(adm_tokens)])) # num of admissions by visit code len (not same yet)
            token_types.append(torch.tensor([adm_token_types]))
            
            # completely tokenize the codes here, from string to ids then to multi-hot
            for i in range(len(self.token_type)):
                # please be note that the type specific vocab is used here
                # is it because the code type is already recorded?
                label_ids = self.tokenizer.convert_tokens_to_ids(adm_masked_labels[i], voc_type=self.token_type_map[i])
                # it seems that different code types have different vocab size and tokenized seperately
                # note that label_hop is just a single multi-hot vector, represeting the masked codes of one type
                label_hop = self._id2multi_hot(label_ids, 
                                               dim=self.tokenizer.token_number(self.token_type_map[i])).unsqueeze(dim=0)
                # masked_labels is 
                # [[diag, shape of n_visit of this patient, diag code vocab total num], 
                # [med, n_visit by num of med code, diag code vocab total num], [pro], [lab]]
                if masked_labels[i] is None:
                    masked_labels[i] = label_hop
                else:
                    masked_labels[i] = torch.cat([masked_labels[i], label_hop], dim=0)  # [num of adm, num of unique codes of this type (vocab size)]
                    
            if len(adm_anomaly_labels) > 0:
                # anomaly detection is to predict the position, while mask prediction is to predict code type (for each type)
                anomaly_labels.append(self._id2multi_hot(adm_anomaly_labels, dim=len(adm_tokens)).unsqueeze(dim=0))
            else:
                anomaly_labels.append(torch.zeros(len(adm_tokens)).unsqueeze(dim=0))
        
        # pad to the same length (add 0 to the right), max length of codes in a visit, of a patient hostory
        visit_positions = torch.tensor(list(range(len(input_tokens))))  # index each admission, not code, [0, 1, 2, 3,...]
        input_tokens = _pad_sequence(input_tokens, pad_id=self.tokenizer.vocab.word2id["[PAD]"]) # here it is still on the id level
        token_types = _pad_sequence(token_types, pad_id=0)
        anomaly_labels = _pad_sequence(anomaly_labels, pad_id=0) if len(anomaly_labels) > 0 else None
        n_adms = len(input_tokens)
        if n_adms > 1:
            # create a fully connected graph between admission (????)
            edge_index = torch.tensor([[i, j] for i in range(n_adms) for j in range(n_adms)]).t()  # [2, n_adms * n_adms]
        else:
            edge_index = torch.tensor([])
            # note that the input_tokens is still at the id level, not embedding level.
            # one-hot method is just for creating labels.
        return input_tokens, token_types, edge_index, visit_positions, masked_labels, anomaly_labels
    
    


class HBERTFinetuneEHRDataset(Dataset):
    def __init__(self, data_pd, tokenizer, token_type, task):
        self.tokenizer = tokenizer
        self.task = task
        
        def transform_data(data, task):
            age_records = {}
            hadm_records = {}  # including current admission and previous admissions
            genders = {}
            labels = {}
            for subject_id in data['SUBJECT_ID'].unique():
                item_df = data[data['SUBJECT_ID'] == subject_id]
                patient, ages = [], []
                
                for _, row in item_df.iterrows():
                    admission = []
                    hadm_id = row['HADM_ID']
                    if "diag" in token_type:
                        admission.append(list(row['ICD9_CODE']))
                    if "med" in token_type:
                        admission.append(list(row['NDC']))
                    if "pro" in token_type:
                        admission.append(list(row['PRO_CODE']))
                    if "lab" in token_type:
                        admission.append(list(row['LAB_TEST'])) # admission of structure[[diag], [med], [pro], [lab]]
                    # patient of structure [[[diag], [med], [pro], [lab]], [[diag], [med], [pro], [lab]], ...]
                    patient.append(admission)
                    # number of ages == number of admissions 
                    ages.append(row['AGE'])
                    
                    # please be note that dataset is retained differently for different tasks
                    if task in ["death", "stay", "readmission"]:  # binary prediction
                        hadm_records[hadm_id] = list(patient)
                        age_records[hadm_id] = ages
                        genders[hadm_id] = [item_df.head(1)["GENDER"].values[0]]
                    # 非常重要:
                    # 为什么 key 是 hadm_id 而不是 subject_id？
                    # 这是因为当前函数的设计目标是将每个入院事件 (HADM_ID) 作为一个预测单元。也就是说：
                    #     •	每个 hadm_id 是一个样本；
                    #     •	对每个 hadm_id，模型需要：
                    #     •	看全部历史（该患者之前的 admissions，包括当前）；
                    #     •	输出一个预测（例如死亡、住院时间、再入院，或者未来 6/12 月是否发生某病）。
                    # 因此，虽然 value = list(patient) 是整个患者的历史，但预测目标绑定的是该 hadm_id 的结局事件，比如当前 admission 是否死亡。   
                        if "READMISSION" in row:
                            labels[hadm_id] = [row["DEATH"], row["STAY_DAYS"], row["READMISSION"]]
                        else:
                            labels[hadm_id] = [row["DEATH"], row["STAY_DAYS"]]
                    else:  # next diagnosis prediction
                        label = row["NEXT_DIAG_6M"] if task == "next_diag_6m" else row["NEXT_DIAG_12M"]
                        if str(label) != "nan":  # only include the admission with next diagnosis
                            # in other words, only admissions with future diagnosis labels are retained.
                            hadm_records[hadm_id] = list(patient)
                            age_records[hadm_id] = ages
                            genders[hadm_id] = [item_df.head(1)["GENDER"].values[0]]
                            labels[hadm_id] = list(label)

            return hadm_records, age_records, genders, labels
        
        self.records, self.ages, self.genders, self.labels = transform_data(data_pd, task)
        
        
    def __len__(self):
        return len(self.records)

    def get_ids(self):
        return list(self.records.keys())
    
    def __getitem__(self, item):
        hadm_id = list(self.records.keys())[item]

        input_tokens, token_types = [], []
        for idx, adm in enumerate(self.records[hadm_id]):  # each subject have multiple admissions, idx:visit id
            adm_tokens = [str(self.ages[hadm_id][idx]) + "_" + self.genders[hadm_id][0]]  # replace [CLS] token with age_gender
            # adm_tokens = [self.ages[hadm_id][idx]]  # replace [CLS] token with age
            adm_token_types = [0]

            for i in range(len(adm)):
                cur_tokens = list(adm[i])
                adm_tokens.extend(cur_tokens)
                adm_token_types.extend([i + 1] * len(cur_tokens)) # 0 for age_gender CLS token
            
            input_tokens.append(adm_tokens)
            token_types.append(adm_token_types)

        if self.task == "death":
            # predict if the patient will die in the hospital
            labels = torch.tensor([self.labels[hadm_id][0]]).float()
        elif self.task == "stay":
            # predict if the patient will stay in the hospital for more than 7 days
            labels = (torch.tensor([self.labels[hadm_id][1]]) > 7).float()
        elif self.task == "readmission":
            # predict if the patient will be readmitted within 1 month
            labels = torch.tensor([self.labels[hadm_id][2]]).float()
        else:
            # predict the next diagnosis in 6 months or 12 months
            # This operation modifies the input, not the label. 
            # It simulates a test-time situation where the model is presented with historical visit data (until the current visit)
            # and a mask token indicating “predict what is in the future”.
            # note that the last (current) visit is not the label, the label is already presented "NEXT_DIAG_12M"
            # please note that this is the only masked token in the finetune dataset,
            # unlike pretrain dataset, where each code type has its own [MASK] token.
            input_tokens[-1] = [input_tokens[-1][0]] + ["[MASK0]"] + input_tokens[-1][1:]
            token_types[-1] = [token_types[-1][0]] + [1] + token_types[-1][1:]
            label_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.labels[hadm_id], 
                                                                          voc_type='diag'))
            labels = torch.zeros(self.tokenizer.token_number(voc_type='diag')).long()
            labels[label_ids] = 1  # multi-hop vector

        visit_positions = torch.tensor(list(range(len(input_tokens))))  # [0, 1, 2, ...]
        input_tokens = [torch.tensor([self.tokenizer.convert_tokens_to_ids(x)]) for x in input_tokens]
        token_types = [torch.tensor([x]) for x in token_types]
        input_tokens = _pad_sequence(input_tokens, pad_id=self.tokenizer.vocab.word2id["[PAD]"])
        token_types = _pad_sequence(token_types, pad_id=0)
        n_adms = len(input_tokens)
        if n_adms > 1:
            # create a fully connected graph between admission
            edge_index = torch.tensor([[i, j] for i in range(n_adms) for j in range(n_adms)]).t()  # [2, n_adms * n_adms]
        else:
            edge_index = torch.tensor([])
            
        return input_tokens, token_types, edge_index, visit_positions, labels
    
# think of it as a function that returns a function, and the returned function only takes batch as input
# which satisfies the DataLoader's collate_fn requirement.
def batcher(pad_id, n_token_type=4, is_train=True):
    def batcher_dev(batch):
        # raw visit position = [0, 1, 2, # of visits - 1]
        raw_input_ids, raw_input_types, raw_edge_indexs, raw_visit_positions, raw_labels = \
            [feat[0] for feat in batch], [feat[1] for feat in batch], [feat[2] for feat in batch], [feat[3] for feat in batch], [feat[4] for feat in batch]

        max_n_tokens = max([x.size(1) for x in raw_input_ids])
        input_ids = torch.cat([F.pad(raw_input_id, (0, max_n_tokens - raw_input_id.size(1)), "constant", pad_id) for raw_input_id in raw_input_ids], dim=0)

        max_n_token_types = max([x.size(1) for x in raw_input_types])
        
        input_types = torch.cat([F.pad(raw_input_type, (0, max_n_token_types - raw_input_type.size(1)), 
                                       "constant", 0) for raw_input_type in raw_input_types], dim=0)
        
        # running sum of the number of patient's visits in a batch
        n_cumsum_nodes = [0] + np.cumsum([input_id.size(0) for input_id in raw_input_ids]).tolist()
        # adjust the edge index to the batch
        # 多个样本的token合并为一个batch时，要调整其图中节点编号以避免冲突；
        # n_cumsum_nodes[i] 是该样本在大batch中token的起始位置，用于“平移”图中的边；
        # 但注意不同病人的visit node不相连。
        edge_index = []
        for i, raw_edge_index in enumerate(raw_edge_indexs):
            if raw_edge_index.shape[0] > 0:
                edge_index.append(raw_edge_index + n_cumsum_nodes[i])
        # this is a within-batch big graph        
        edge_index = torch.cat(edge_index, dim=1) if len(edge_index) > 0 else None

        # simply concatenating all these visit indices into a single 1D tensor for the whole batch.
        visit_positions = torch.cat(raw_visit_positions, dim=0)

        if is_train:
            labels = []  # [n_token_type, B, n_tokens (of that code type)], 
            # each element is a multi-hop label tensor
            for i in range(n_token_type):
                labels.append(torch.cat([x[i] for x in raw_labels], dim=0))

            raw_anomaly_labels = [feat[5] for feat in batch]
            if raw_anomaly_labels[0] is not None:
                # anomaly_labels is also padded in the dataset object (n_visit of one patient by max patient code len), 
                # so we need to pad it here as well at the batch level
                max_n_anomaly_labels = max([x.size(1) for x in raw_anomaly_labels])
                anomaly_labels = torch.cat([F.pad(raw_anomaly_label, (0, max_n_anomaly_labels - raw_anomaly_label.size(1)), "constant", 0) \
                    for raw_anomaly_label in raw_anomaly_labels], dim=0)
            else:
                anomaly_labels = None
            return input_ids, input_types, edge_index, visit_positions, labels, anomaly_labels
        else:
            labels = torch.stack(raw_labels, dim=0)
            labeled_batch_idx = [n - 1 for n in n_cumsum_nodes[1:]]  # indicate the index of the to-be-predicted admission (last visit of each patient)
            return input_ids, input_types, edge_index, visit_positions, labeled_batch_idx, labels
    
    return batcher_dev


class UniqueIDSampler(Sampler):
    def __init__(self, ids, batch_size, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.ids = ids
        self.num_samples = len(ids)
        self.index_id_map = list(enumerate(ids))  # [(idx, id), ...]

    def __iter__(self):
        remaining = self.index_id_map[:]
        random.shuffle(remaining)

        batch, used_ids = [], set()

        while remaining:
            idx, uid = remaining.pop(0)
            if uid not in used_ids:
                batch.append(idx)
                used_ids.add(uid)
            else:
                remaining.append((idx, uid))  # push back to queue

            if len(batch) == self.batch_size:
                yield batch
                batch, used_ids = [], set()

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        est_batches = self.num_samples // self.batch_size
        if not self.drop_last and self.num_samples % self.batch_size != 0:
            est_batches += 1
        return est_batches