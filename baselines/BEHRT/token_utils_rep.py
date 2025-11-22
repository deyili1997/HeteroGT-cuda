from typing import Union
import numpy as np
import torch
from tree_utils_rep import build_atc_tree, build_icd9_tree

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
    def __init__(self, diag_sentences, gender_set, age_set, age_gender_set, special_tokens):
        self.vocab = Voc() # this is the global vocabulary
        
        # Add special tokens to the global voc table. 
        # Note that the PAD is 0.
        self.vocab.add_sentence(special_tokens)
        self.n_special_tokens = len(special_tokens)
        # create specific vocabularies for each type of code
        # at the same time, update the global vocabulary
        self.age_voc = self.add_vocab(age_set)
        self.diag_voc = self.add_vocab(diag_sentences)
        self.gender_voc = self.add_vocab(gender_set)
        self.age_gender_voc = self.add_vocab(age_gender_set)
        
        assert len(special_tokens) + len(self.age_voc.id2word) + \
            len(self.diag_voc.id2word) + len(self.gender_voc.id2word) + \
            len(self.age_gender_voc.id2word) == len(self.vocab.id2word)

    def add_vocab(self, sentences):
        voc = self.vocab
        specific_voc = Voc()
        for sentence in sentences:
            # be note that the global vocab and the specific vocab are different
            # they are updated separately here
            voc.add_sentence(sentence)
            specific_voc.add_sentence(sentence)
        return specific_voc 
        
    def convert_tokens_to_ids(self, tokens, voc_type="all"):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if voc_type == "all":
                ids.append(self.vocab.word2id[token])
            elif voc_type == "diag":
                ids.append(self.diag_voc.word2id[token])
        return ids

    def convert_ids_to_tokens(self, ids, voc_type="all"):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            if voc_type == "all":
                tokens.append(self.vocab.id2word[i])
            elif voc_type == "diag":
                tokens.append(self.diag_voc.id2word[i])
        return tokens
    
    def token_id_range(self, voc_type="diag"):
        age_size = len(self.age_voc.id2word)
        diag_size = len(self.diag_voc.id2word)

        if voc_type == "diag":
            return [self.n_special_tokens + age_size, self.n_special_tokens + age_size + diag_size]
    
    def token_number(self, voc_type="diag"):
        if voc_type == "diag":
            return len(self.diag_voc.id2word)

    
    def random_token(self, voc_type="diag"):
        # randomly sample a token from the vocabulary
        if voc_type == "diag":
            return self.diag_voc.id2word[np.random.randint(len(self.diag_voc.id2word))]
    