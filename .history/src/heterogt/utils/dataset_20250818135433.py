
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def _id2multi_hot(ids, dim):
    # return a multi-hot vector of size dim, where ids are the indices to be set to 1
    multi_hot = torch.zeros(dim, dtype=torch.long, device=ids.device)
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
                        # genders[hadm_id] = row["GENDER"]
                        ages[hadm_id] = list(age)
                        labels[hadm_id] = [row["DEATH"], row["STAY_DAYS"], row["READMISSION"]]
                    else: # next diagnosis prediction
                        label = row["NEXT_DIAG_6M"] if task == "next_diag_6m" else row["NEXT_DIAG_12M"]
                        if str(label) != "nan":  # only include the admission with next diagnosis
                            hadm_records[hadm_id] = list(patient) # this is why this line is inside the if statement
                            # genders[hadm_id] = row["GENDER"]
                            ages[hadm_id] = list(age)
                            labels[hadm_id] = list(label)
            return hadm_records, genders, ages, labels
        self.records, self.genders, self.ages, self.labels = transform_data(ehr_finetune_data, task)

    def __len__(self):
        return len(self.records)
    
    def get_ids(self):
        return list(self.records.keys())
    
    def __getitem__(self, idx): # return tokenized input_ids, token_types, adm_index, age_sex, labels
        hadm_id = list(self.records.keys())[idx]
        input_tokens = []
        token_types = [] 
        adm_index = []
        # gender = self.genders[hadm_id] # single value
        ages = self.ages[hadm_id] # list of ages, one for each admission
        age_genders = []

        # iterate through all encounters in this encounter sequence
        for idx, adm in enumerate(self.records[hadm_id]):
            adm_tokens = []
            adm_token_types = []
            # iterate through all types of tokens in this encounter
            for i in range(len(adm)):
                cur_tokens = list(adm[i])
                adm_tokens.extend(cur_tokens)
                adm_token_types.extend([i + 1] * len(cur_tokens)) # we have 0 for PAD token
                
            input_tokens.extend(adm_tokens)
            token_types.extend(adm_token_types)
            # 0 is used for PAD token
            adm_index.extend([idx + 1] * len(adm_tokens))
            age_genders.append(ages[idx])

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
        age_gender_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(age_genders, voc_type="all")], dtype=torch.long)

        # sanity check
        assert input_ids.shape == token_types.shape == adm_index.shape, \
            f"Input IDs shape {input_ids.shape}, token types shape {token_types.shape}, adm index shape {adm_index.shape} do not match"
        assert age_gender_ids.size(1) == len(self.records[hadm_id]), \
            f"Age gender index length {age_gender_ids.size(1)} does not match input IDs length {len(self.records[hadm_id])}"
        return input_ids, token_types, adm_index, age_gender_ids, labels


def batcher(tokenizer, task_index, n_token_type=4, is_pretrain=False):
    def batcher_dev(batch):
        raw_input_ids, raw_token_types, raw_adm_index, raw_age_gender_index, raw_labels = \
            [feat[0] for feat in batch], [feat[1] for feat in batch], [feat[2] for feat in batch], [feat[3] for feat in batch], [feat[4] for feat in batch]

        seq_len = torch.tensor([x.size(1) for x in raw_input_ids])
        adm_num = torch.tensor([x.size(1) for x in raw_age_gender_index])
        max_n_tokens = seq_len.max().item()
        max_n_adms = adm_num.max().item()
        
        seq_pad_id = tokenizer.convert_tokens_to_ids(["[PAD]"], voc_type="all")[0]
        token_pad_id = 0
        adm_pad_id = 0
        age_gender_pad_id = tokenizer.convert_tokens_to_ids(["[PAD]"], voc_type="all")[0]

        input_ids = torch.cat([F.pad(raw_input_id, (0, max_n_tokens - raw_input_id.size(1)), "constant", seq_pad_id) for raw_input_id in raw_input_ids], dim=0)
        token_types = torch.cat([F.pad(raw_token_type, (0, max_n_tokens - raw_token_type.size(1)), "constant", token_pad_id) for raw_token_type in raw_token_types], dim=0)
        adm_index = torch.cat([F.pad(raw_adm_idx, (0, max_n_tokens - raw_adm_idx.size(1)), "constant", adm_pad_id) for raw_adm_idx in raw_adm_index], dim=0)
        age_gender_ids = torch.cat([F.pad(raw_age_gender_idx, (0, max_n_adms - raw_age_gender_idx.size(1)), "constant", age_gender_pad_id) for raw_age_gender_idx in raw_age_gender_index], dim=0)
        assert input_ids.shape == token_types.shape == adm_index.shape

        if is_pretrain:
            labels = []  # [n_token_type, B, n_tokens (of that code type)], 
            # each element is a multi-hop label tensor
            for i in range(n_token_type):
                labels.append(torch.cat([x[i] for x in raw_labels], dim=0))
            # convert to float tensor
            for i in range(len(labels)):
                labels[i] = labels[i].float()
        else:
            labels = torch.stack(raw_labels, dim=0).float()
        
        # sanity check
        for row in range(input_ids.size(0)):
            count_ids   = (input_ids[row]   != 0).sum().item()
            count_types = (token_types[row] != 0).sum().item()
            count_adm   = (adm_index[row]   != 0).sum().item()
            expected    = seq_len[row].item()

            assert count_ids == count_types == count_adm == expected, \
                (f"Row {row}: counts mismatch. "
                f"ids={count_ids}, types={count_types}, adm={count_adm}, expected seq_len={expected}")

        for row in range(age_gender_ids.size(0)):
            count_age_gender = (age_gender_ids[row] != 0).sum().item()
            expected = adm_num[row].item()
            assert count_age_gender == expected, \
                (f"Row {row}: counts mismatch. "
                 f"age_gender={count_age_gender}, expected adm_num={expected}")

        return input_ids, token_types, adm_index, age_gender_ids, task_index, labels

    return batcher_dev