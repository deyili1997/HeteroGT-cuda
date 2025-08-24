import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random

def expand_level3():
    # map each ICD-9 code to a range
    # return 
    # {
        #   '001': '001-139',
        #   '002': '001-139',
        #   ...
        #   '250': '240-279',
    # }
    
    level3 = ["001-139", "140-239", "240-279", "280-289", "290-319", "320-389", "390-459", "460-519",
              "520-579", "580-629", "630-679", "680-709", "710-739", "740-759", "760-779", "780-799",
              "800-999", "E000-E999", "V01-V91"]

    level3_expand = {}
    for i in level3:
        tokens = i.split('-')
        if i[0] == 'V':
            if len(tokens) == 1:
                level3_expand[i] = i
            else:
                for j in range(int(tokens[0][1:]), int(tokens[1][1:]) + 1):
                    level3_expand["V%02d" % j] = i
        elif i[0] == 'E':
            if len(tokens) == 1:
                level3_expand[i] = i
            else:
                for j in range(int(tokens[0][1:]), int(tokens[1][1:]) + 1):
                    level3_expand["E%03d" % j] = i
        else:
            if len(tokens) == 1:
                level3_expand[i] = i
            else:
                for j in range(int(tokens[0]), int(tokens[1]) + 1):
                    level3_expand["%03d" % j] = i
    return level3_expand, level3

def _id2multi_hot(ids, dim):
    # 判断 ids 是 tensor 还是 list
    if isinstance(ids, torch.Tensor):
        device = ids.device
    else:
        device = "cpu"  # list/tuple 时默认在 CPU 上

    multi_hot = torch.zeros(dim, dtype=torch.long, device=device)
    multi_hot[ids] = 1
    return multi_hot

class PreTrainEHRDataset(Dataset):
    def __init__(self, ehr_pretrain_data, tokenizer, token_type, mask_rate, group_code_thre, max_num_adms):
        self.tokenizer = tokenizer
        self.token_type = token_type
        assert mask_rate > 0, "Mask rate must be greater than 0"
        self.mask_rate = mask_rate
        self.level4_dict, _ = expand_level3()
        self.token_type_id_dict = {
            "[PAD]": 0,
            "[CLS]": 1,
            "diag": 2,
            "med": 3,
            "lab": 4,
            "pro": 5,
            "group": 6,
            "visit": 7
        }
        
        self._init_common_attributes(group_code_thre, max_num_adms)
        
        def _transform_pretrain_data(data):
            hadm_records = {}  # including current admission and previous admissions
            ages = {}
            hadm_types = {}
            for subject_id in data['SUBJECT_ID'].unique():
                item_df = data[data['SUBJECT_ID'] == subject_id]
                patient, age, adm_type = [], [], []
                for _, row in item_df.iterrows():
                    admission = []
                    if "diag" in token_type:
                        admission.append(list(row['ICD9_CODE']))
                    if "med" in token_type:
                        admission.append(list(row['NDC']))
                    if "lab" in token_type:
                        admission.append(list(row['LAB_TEST']))
                    if "pro" in token_type:
                        admission.append(list(row['PRO_CODE']))
                    patient.append(admission)
                    age.append(row['AGE'])
                    adm_type.append(row['ADMISSION_TYPE'])
                    
                hadm_records[subject_id] = list(patient)
                ages[subject_id] = list(age)
                hadm_types[subject_id] = list(adm_type)
            return hadm_records, ages, hadm_types
        self.records, self.ages, self.adm_types = _transform_pretrain_data(ehr_pretrain_data)

    def _init_common_attributes(self, group_code_thre, max_num_adms):
        """初始化共同属性的辅助方法"""
        self.group_code_thre = group_code_thre
        self.max_num_adms = max_num_adms
        self.group_code_adm_index = max_num_adms + 1
        self.group_code_type_id = self.token_type_id_dict["group"]
        self.cls_adm_index = max_num_adms + 2
        self.cls_type_id = self.token_type_id_dict["[CLS]"]

    def __len__(self):
        return len(self.records)
        
    def get_ids(self):
        return list(self.records.keys())
        
    def __getitem__(self, idx):
        subject_id = list(self.records.keys())[idx]
        input_tokens = ['[CLS]']
        token_types = [self.cls_type_id]  # token type for CLS token
        adm_index = [self.cls_adm_index]
        ages = self.ages[subject_id] # list of ages, one for each admission
        adm_types = self.adm_types[subject_id] # list of adm types, one for each admission
        diag_group_codes = {}
        masked_labels = [None for _ in range(len(self.token_type))]
        
        # Step 1
        global_code_pool = {code_type: set() for code_type in self.token_type}
        for adm in self.records[subject_id]:
            for i, codes in enumerate(adm):
                global_code_pool[self.token_type[i]].update(codes)

        # Step 2
        masked_code_pool = {}
        for code_type in self.token_type:
            all_codes = list(global_code_pool[code_type])
            n_mask = max(1, int(len(all_codes) * self.mask_rate))
            masked_codes = np.random.choice(all_codes, size=n_mask, replace=False)
            masked_code_pool[code_type] = set(masked_codes)
        
        # Step 3: encode visits and build input
        curr_pos = 1 # we already have CLS token
        # iterate through all encounters in this encounter sequence
        for idx, adm in enumerate(self.records[subject_id]):
            adm_tokens = []
            adm_token_types = []
            # iterate through all types of tokens in this encounter
            for i in range(len(adm)):
                cur_tokens = list(adm[i])
                code_type = self.token_type[i]
                # pre-training task: mask some tokens
                mask_set = masked_code_pool[code_type]
                non_masked_tokens = [tok for tok in cur_tokens if tok not in mask_set]
                
                # important: 兜底防空token list：放回当前就诊该类型的 1 个 token
                if len(non_masked_tokens) == 0 and len(cur_tokens) > 0:
                    keep_tok = random.choice(cur_tokens)
                    non_masked_tokens = [keep_tok]
                    
                if i == 0: # if it is diag codes
                    diag_group_codes = self.update_group_code_dict(curr_pos, diag_group_codes, non_masked_tokens)
                    # here we predict the ontology of the masked diag codes
                adm_tokens.extend(non_masked_tokens)
                adm_token_types.extend([i + 2] * len(non_masked_tokens)) # +2 to distinguish from PAD and CLS tokens
                curr_pos += len(non_masked_tokens)
            
            input_tokens.extend(adm_tokens)
            token_types.extend(adm_token_types)
            adm_index.extend([idx + 1] * len(adm_tokens)) # 0 is used for PAD token
        
        #we append the group code at the end
        diag_group_codes = self.filter_dict(diag_group_codes, self.group_code_thre, input_tokens)
        group_code_tokens = list(diag_group_codes.keys())
        input_tokens.extend(group_code_tokens)
        token_types.extend([self.group_code_type_id] * len(group_code_tokens))
        adm_index.extend([self.group_code_adm_index] * len(group_code_tokens))
        diag_group_codes = self.reindex_dict(diag_group_codes, curr_pos)  # reindex the group codes to start from curr_pos, which means append to the tail of the sequence 
        
        # Step 5: build final labels directly from masked_code_pool
        for i in range(len(self.token_type)):
            code_type = self.token_type[i]
            label_ids = self.tokenizer.convert_tokens_to_ids(
                list(masked_code_pool[code_type]),
                voc_type=code_type
            )
            label_vec = _id2multi_hot(label_ids, dim=self.tokenizer.token_number(self.token_type[i]))
            masked_labels[i] = label_vec.long().view(1, -1)

        # build group code label for masked diag codes
        group_code_label_tokens = [self.find_level4_code(token) for token in masked_code_pool['diag'] if self.find_level4_code(token) is not None] # collect for pretrain task
        group_code_label_id = self.tokenizer.convert_tokens_to_ids(group_code_label_tokens, voc_type = 'group')
        group_code_label_vec = _id2multi_hot(group_code_label_id, dim=self.tokenizer.token_number('group'))
        group_code_labels = group_code_label_vec.long()
        
        # build adm types label
        adm_type_ids = self.tokenizer.convert_tokens_to_ids(adm_types, voc_type="adm_type")
        adm_type_label_vec = _id2multi_hot(adm_type_ids, dim=self.tokenizer.token_number('adm_type'))
        adm_type_labels = adm_type_label_vec.long()

        # Step 6: convert input_tokens to ids
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(input_tokens, voc_type = "all")], dtype=torch.long)
        # convert token_types to tensor
        token_types = torch.tensor([token_types], dtype=torch.long)
        # convert adm_index to tensor
        adm_index = torch.tensor([adm_index], dtype=torch.long)
        # convert age_index to tensor
        age_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(ages, voc_type="all")], dtype=torch.long) 
        self.sanity_check(subject_id, input_ids, token_types, adm_index, age_ids, diag_group_codes)
        return input_ids, token_types, adm_index, age_ids, diag_group_codes, {'masked': masked_labels, 'group': group_code_labels, 'adm_type': adm_type_labels}

    def sanity_check(self, id, input_ids, token_types, adm_index, age_ids, diag_group_codes):
        # sanity check
        assert input_ids.shape == token_types.shape == adm_index.shape, \
            f"Input IDs shape {input_ids.shape}, token types shape {token_types.shape}, adm index shape {adm_index.shape} do not match"
        assert age_ids.size(1) == len(self.records[id]), \
            f"Age index length {age_ids.size(1)} does not match input IDs length {len(self.records[id])}"
        assert torch.unique(adm_index[0][(adm_index[0] > 0) & (adm_index[0] <= self.max_num_adms)]).numel() == age_ids.size(1), \
            f"Admission unique count {torch.unique(adm_index[0][(adm_index[0] > 0) & (adm_index[0] <= self.max_num_adms)]).numel()} does not match age index size {age_ids.size(1)}"

        all_diag_pos = sum(diag_group_codes.values(), [])
        idx = torch.tensor(all_diag_pos, dtype=torch.long, device=token_types.device)
        assert (idx < token_types[0].size(0)).all(), "some idx out of range"
        vals = token_types[0][idx]
        assert torch.all(vals == self.token_type_id_dict['diag']), f"These pos are not diag type id: { [p for p, v in zip(all_diag_pos, vals.tolist()) if v != self.token_type_id_dict['diag']] }"
        all_group_code_pos = list(diag_group_codes.keys())
        group_code_idx = torch.tensor(all_group_code_pos, dtype=torch.long, device=token_types.device)
        group_code_type_ids = token_types[0][group_code_idx]
        group_code_adm_index = adm_index[0][group_code_idx]
        assert torch.all(group_code_type_ids == self.group_code_type_id), \
        f"These group code pos are not group code type id: { [p for p, v in zip(all_group_code_pos, group_code_type_ids.tolist()) if v != self.group_code_type_id] }"
        assert torch.all(group_code_adm_index == self.group_code_adm_index), \
            f"These group code adm index are not {self.group_code_adm_index}: { [p for p, v in zip(all_group_code_pos, group_code_adm_index.tolist()) if v != self.group_code_adm_index] }"
        assert (len(diag_group_codes) == (token_types[0] == self.group_code_type_id).sum().item()), \
            f"Group code count {len(diag_group_codes)} does not match token type count { (token_types[0] == self.group_code_type_id).sum().item()}"
        group_code_id_range = self.tokenizer.token_id_range("group")
        group_code_ids = input_ids[0][group_code_idx]
        assert (group_code_ids >= group_code_id_range[0]).all() and (group_code_ids <= group_code_id_range[1]).all(), \
            f"Group code IDs {group_code_ids.tolist()} are out of range {group_code_id_range.tolist()}"

    def update_group_code_dict(self, curr_pos, diag_group_codes, diag_codes):
        pos_count = 0
        for code in diag_codes:
            level4 = self.find_level4_code(code)
            if level4:
                if level4 not in diag_group_codes:
                    diag_group_codes[level4] = []
                diag_group_codes[level4].append(curr_pos + pos_count)
            pos_count += 1
        return diag_group_codes
    
    def find_level4_code(self, code):
        level1 = code[5:] # remove the "DIAG_"
        level2 = level1[:4] if level1[0] == 'E' else level1[:3]
        level4 = self.level4_dict[level2] if level2 in self.level4_dict else None
        return level4
    
    def filter_dict(self, d, threshold, input_tokens):
        # only keep group code with enough diag codes
        filtered_dict = {}
        for group_token, pos in d.items():
            # 从 list 中取出对应位置的 tokens
            tokens = [input_tokens[i] for i in pos]
            # 用 set 计算唯一值数量
            uni_tokens_num = len(set(tokens))
            if uni_tokens_num >= threshold:
                filtered_dict[group_token] = pos
        return filtered_dict

    def reindex_dict(self, d, current_index):
        return {new_k: v for new_k, v in zip(range(current_index, current_index + len(d)), d.values())}
          
class FineTuneEHRDataset(PreTrainEHRDataset):
    def __init__(self, ehr_finetune_data, tokenizer, token_type, task, max_num_adms, group_code_thre):
        super().__init__(ehr_finetune_data, tokenizer, token_type, mask_rate=0.1, 
                         group_code_thre=group_code_thre, max_num_adms=max_num_adms)
        self.task = task
        
        def _transform_finetune_data(data, task):
            hadm_records = {}
            ages = {}
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
                        ages[hadm_id] = list(age)
                        labels[hadm_id] = [row["DEATH"], row["STAY_DAYS"], row["READMISSION"]]
                    else: # next diagnosis prediction
                        label = row["NEXT_DIAG_6M"] if task == "next_diag_6m" else row["NEXT_DIAG_12M"]
                        if str(label) != "nan":  # only include the admission with next diagnosis
                            hadm_records[hadm_id] = list(patient) # this is why this line is inside the if statement
                            ages[hadm_id] = list(age)
                            labels[hadm_id] = list(label)
            return hadm_records, ages, labels

        self.records, self.ages, self.labels = _transform_finetune_data(ehr_finetune_data, self.task)
    
    def __getitem__(self, idx): # return tokenized input_ids, token_types, adm_index, age_sex, labels
        hadm_id = list(self.records.keys())[idx]
        input_tokens = ['[CLS]']
        token_types = [self.cls_type_id]  # token type for CLS token
        adm_index = [self.cls_adm_index]
        ages = self.ages[hadm_id] # list of ages, one for each admission
        diag_group_codes = {}
        
        curr_pos = 1 # we already have CLS token
        # iterate through all encounters in this encounter sequence
        for idx, adm in enumerate(self.records[hadm_id]):
            adm_tokens = []
            adm_token_types = []
            # iterate through all types of tokens in this encounter
            for i in range(len(adm)):
                cur_tokens = list(adm[i])
                if i == 0: # if it is diag codes
                    diag_group_codes = self.update_group_code_dict(curr_pos, diag_group_codes, cur_tokens)
                adm_tokens.extend(cur_tokens)
                adm_token_types.extend([i + 2] * len(cur_tokens)) # [PAD]: 0, [CLS]: 1, diag: 2, med: 3, lab: 4, pro: 5, group: 6, visit: 7
                curr_pos += len(cur_tokens)

            input_tokens.extend(adm_tokens)
            token_types.extend(adm_token_types)
            adm_index.extend([idx + 1] * len(adm_tokens)) # 0 is used for PAD token
        
        # finally, we append the group code at the end
        diag_group_codes = self.filter_dict(diag_group_codes, self.group_code_thre, input_tokens)
        input_tokens.extend(list(diag_group_codes.keys()))
        token_types.extend([self.group_code_type_id] * len(diag_group_codes))
        adm_index.extend([self.group_code_adm_index] * len(diag_group_codes))
        diag_group_codes = self.reindex_dict(diag_group_codes, curr_pos)  # reindex the group codes to start from curr_pos, which means append to the tail of the sequence

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
        # convert age_index to tensor
        age_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(ages, voc_type="all")], dtype=torch.long)

        self.sanity_check(hadm_id, input_ids, token_types, adm_index, age_ids, diag_group_codes)

        return input_ids, token_types, adm_index, age_ids, diag_group_codes, labels


def batcher(tokenizer, n_token_type=4, is_pretrain=False):
    def batcher_dev(batch):
        raw_input_ids, raw_token_types, raw_adm_index, raw_age_index, raw_diag_code_group_dicts, raw_labels = \
            [feat[0] for feat in batch], [feat[1] for feat in batch], [feat[2] for feat in batch], [feat[3] for feat in batch], [feat[4] for feat in batch], [feat[5] for feat in batch]

        seq_len = torch.tensor([x.size(1) for x in raw_input_ids])
        adm_num = torch.tensor([x.size(1) for x in raw_age_index])
        max_n_tokens = seq_len.max().item()
        max_n_adms = adm_num.max().item()
        
        seq_pad_id = tokenizer.convert_tokens_to_ids(["[PAD]"], voc_type="all")[0]
        token_type_pad_id = 0
        adm_index_pad_id = 0
        age_pad_id = tokenizer.convert_tokens_to_ids(["[PAD]"], voc_type="all")[0]

        input_ids = torch.cat([F.pad(raw_input_id, (0, max_n_tokens - raw_input_id.size(1)), "constant", seq_pad_id) for raw_input_id in raw_input_ids], dim=0)
        token_types = torch.cat([F.pad(raw_token_type, (0, max_n_tokens - raw_token_type.size(1)), "constant", token_type_pad_id) for raw_token_type in raw_token_types], dim=0)
        adm_index = torch.cat([F.pad(raw_adm_idx, (0, max_n_tokens - raw_adm_idx.size(1)), "constant", adm_index_pad_id) for raw_adm_idx in raw_adm_index], dim=0)
        age_ids = torch.cat([F.pad(raw_age_idx, (0, max_n_adms - raw_age_idx.size(1)), "constant", age_pad_id) for raw_age_idx in raw_age_index], dim=0)
        assert input_ids.shape == token_types.shape == adm_index.shape

        if is_pretrain:
            masked_labels = [torch.cat([x['masked'][i] for x in raw_labels], dim=0).float() for i in range(n_token_type)]
            group_labels = torch.stack([x['group'] for x in raw_labels], dim=0).float()
            adm_type_labels = torch.stack([x['adm_type'] for x in raw_labels], dim=0).float()
            labels = {'masked': masked_labels, 'group': group_labels, 'adm_type': adm_type_labels}
        else:
            labels = torch.stack(raw_labels, dim=0).float()
        
        # sanity check
        for row in range(input_ids.size(0)):
            count_ids = (input_ids[row] != 0).sum().item()
            count_types = (token_types[row] != 0).sum().item()
            count_adm = (adm_index[row] != 0).sum().item()
            expected = seq_len[row].item()

            assert count_ids == count_types == count_adm == expected, \
                (f"Row {row}: counts mismatch. "
                f"ids={count_ids}, types={count_types}, adm={count_adm}, expected seq_len={expected}")

        for row in range(age_ids.size(0)):
            count_age = (age_ids[row] != 0).sum().item()
            expected = adm_num[row].item()
            assert count_age == expected, \
                (f"Row {row}: counts mismatch. "
                 f"age={count_age}, expected adm_num={expected}")
        return input_ids, token_types, adm_index, age_ids, raw_diag_code_group_dicts, labels
    return batcher_dev