from typing import Union
import numpy as np

class Voc(object):
    def __init__(self):
        self.id2word = {}
        self.word2id = {}

    def add_sentence(self, sentence: Union[list, np.ndarray]):
        # sentence is a list of codes
        for word in sentence:
            if word not in self.word2id:
                self.id2word[len(self.word2id)] = word
                self.word2id[word] = len(self.word2id)
                
class EHRTokenizer(object):
    def __init__(self, age_sentences, group_code_sentences, diag_sentences, med_sentences, lab_sentences, pro_sentences, special_tokens):
        self.vocab = Voc() # this is the global vocabulary, not including age, token type, adm_index and task

        # Add special tokens to the global voc table. 
        # Note that the PAD is 0.
        self.vocab.add_sentence(special_tokens)
        self.n_special_tokens = len(special_tokens)
        # create specific vocabularies for each type of code
        # at the same time, update the global vocabulary
        # share the global vocab
        self.age_voc = self.add_vocab(age_sentences)
        self.group_code_voc = self.add_vocab(group_code_sentences)
        self.diag_voc = self.add_vocab(diag_sentences)
        self.med_voc = self.add_vocab(med_sentences)
        self.lab_voc = self.add_vocab(lab_sentences)
        self.pro_voc = self.add_vocab(pro_sentences)

        assert len(special_tokens) + len(self.age_voc.id2word) + len(self.group_code_voc.id2word) + \
            len(self.diag_voc.id2word) + len(self.med_voc.id2word) + len(self.lab_voc.id2word) + len(self.pro_voc.id2word) == len(self.vocab.id2word)

    def add_vocab(self, sentences):
        voc = self.vocab
        specific_voc = Voc()
        for sentence in sentences:
            # be note that the global vocab and the specific vocab are different
            # they are updated separately here, if global_update is True,
            voc.add_sentence(sentence)
            specific_voc.add_sentence(sentence)
        return specific_voc 
        
    def convert_tokens_to_ids(self, tokens, voc_type="all"):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if voc_type == "all":
                ids.append(self.vocab.word2id[token])
            elif voc_type == "group":
                ids.append(self.group_code_voc.word2id[token])
            elif voc_type == "diag":
                ids.append(self.diag_voc.word2id[token])
            elif voc_type == "med":
                ids.append(self.med_voc.word2id[token])
            elif voc_type == "lab":
                ids.append(self.lab_voc.word2id[token])
            elif voc_type == "pro":
                ids.append(self.pro_voc.word2id[token])
            elif voc_type == "age":
                ids.append(self.age_voc.word2id[token])
            else:
                raise ValueError(f"Unknown vocabulary type: {voc_type}")
        return ids

    def convert_ids_to_tokens(self, ids, voc_type="all"):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            if voc_type == "all":
                tokens.append(self.vocab.id2word[i])
            elif voc_type == "group":
                tokens.append(self.group_code_voc.id2word[i])
            elif voc_type == "diag":
                tokens.append(self.diag_voc.id2word[i])
            elif voc_type == "med":
                tokens.append(self.med_voc.id2word[i])
            elif voc_type == "lab":
                tokens.append(self.lab_voc.id2word[i])
            elif voc_type == "pro":
                tokens.append(self.pro_voc.id2word[i])
            elif voc_type == "age":
                tokens.append(self.age_voc.id2word[i])
            else:
                raise ValueError(f"Unknown vocabulary type: {voc_type}")
        return tokens
    
    def token_id_range(self, voc_type="diag"):
        age_size = len(self.age_voc.id2word)
        group_code_size = len(self.group_code_voc.id2word)
        diag_size = len(self.diag_voc.id2word)
        med_size = len(self.med_voc.id2word)
        lab_size = len(self.lab_voc.id2word)
        if voc_type == "special":
            return [0, self.n_special_tokens]
        elif voc_type == "age":
            return [self.n_special_tokens, self.n_special_tokens + age_size]
        elif voc_type == "group":
            return [self.n_special_tokens + age_size, self.n_special_tokens + age_size + group_code_size]
        elif voc_type == "diag":
            return [self.n_special_tokens + age_size + group_code_size, self.n_special_tokens + age_size + group_code_size + diag_size]
        elif voc_type == "med":
            return [self.n_special_tokens + age_size + group_code_size + diag_size, self.n_special_tokens + age_size + group_code_size + diag_size + med_size]
        elif voc_type == "lab":
            return [self.n_special_tokens + age_size + group_code_size + diag_size + med_size, self.n_special_tokens + age_size + group_code_size + diag_size + med_size + lab_size]
        elif voc_type == "pro":
            return [self.n_special_tokens + age_size + group_code_size + diag_size + med_size + lab_size, len(self.vocab.id2word)]
        # age is not part of the global vocab, it is just for embedding

    def token_number(self, voc_type="diag"):
        if voc_type == "all":
            return len(self.vocab.id2word)
        elif voc_type == "age":
            return len(self.age_voc.id2word)
        elif voc_type == "group":
            return len(self.group_code_voc.id2word)
        elif voc_type == "diag":
            return len(self.diag_voc.id2word)
        elif voc_type == "med":
            return len(self.med_voc.id2word)
        elif voc_type == "lab":
            return len(self.lab_voc.id2word)
        elif voc_type == "pro":
            return len(self.pro_voc.id2word)
        elif voc_type == "age":
            return len(self.age_voc.id2word)
        else:
            raise ValueError(f"Unknown vocabulary type: {voc_type}")

    def random_token(self, voc_type="diag"):
        # randomly sample a token from the vocabulary
        if voc_type == "diag":
            return self.diag_voc.id2word[np.random.randint(len(self.diag_voc.id2word))]
        elif voc_type == "med":
            return self.med_voc.id2word[np.random.randint(len(self.med_voc.id2word))]
        elif voc_type == "lab":
            return self.lab_voc.id2word[np.random.randint(len(self.lab_voc.id2word))]
        elif voc_type == "pro":
            return self.pro_voc.id2word[np.random.randint(len(self.pro_voc.id2word))]   
        elif voc_type == "age":
            return self.age_voc.id2word[np.random.randint(len(self.age_voc.id2word))]
        elif voc_type == "group":
            return self.group_code_voc.id2word[np.random.randint(len(self.group_code_voc.id2word))]
        else:
            raise ValueError(f"Unknown vocabulary type: {voc_type}")
