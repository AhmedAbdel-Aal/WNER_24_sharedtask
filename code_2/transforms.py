import torch
from transformers import BertTokenizer
from functools import partial
import re
import itertools
import datasets


class BertSeqTransform:
    def __init__(self, bert_model, label_map, max_seq_len=512):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.encoder = partial(
            self.tokenizer.encode,
            max_length=max_seq_len,
            truncation=True,
        )
        self.max_seq_len = max_seq_len
        self.label_map = label_map
        print(label_map)
        # pad by an id to be ignored --> -100 (ignored in pytorch)
        self.pad_token = datasets.Token(text="UNK")
        self.pad_token_id = torch.nn.CrossEntropyLoss().ignore_index
        # SEP token & id
        self.sep_token = self.tokenizer.sep_token
        self.sep_token_id = self.tokenizer.sep_token_id
        # CLS token & id
        self.cls_token = self.tokenizer.cls_token
        self.cls_token_id = self.tokenizer.cls_token_id
        # tag of label O
        self.id_of_o = self.label_map["O"]
        #
        

    def __call__(self, segment):
        input_ids, tags, tokens = list(), list(), list()
        #print('hi')
        #print(self.vocab.tags[0].get_stoi())
        #print('tag of O: ', self.tag_of_o)
        

        for token in segment:
            # Sometimes the tokenizer fails to encode the word and return no input_ids, in that case, we use the input_id for [UNK]
            token_input_ids = self.encoder(token.text)[1:-1] or self.pad_token_id
            input_ids += token_input_ids
            # append tags with the tag of O(outside) if the word has mutliple tokens
            tags += [self.label_map[token.gold_tag[0]]] #actual tag
            tags += [self.id_of_o] * (len(token_input_ids) - 1) #tag of O
            # append tokens with UNK if the word has mutliple tokens
            tokens += [token] + [self.pad_token] * (len(token_input_ids) - 1)
        
        # Truncate to max_seq_len if needed
        truncat_length = (len(input_ids) + 2) - self.max_seq_len
        if truncat_length > 0:
            text = " ".join([t.text for t in tokens if t.text != "UNK"])
            print("Truncating the sequence %s to %d", text, self.max_seq_len - 2)
            input_ids = input_ids[:self.max_seq_len - 2]
            tags = tags[:self.max_seq_len - 2]
            tokens = tokens[:self.max_seq_len - 2]

        # create attention mask
        attention_mask = [1] * len(input_ids)
        

        # add the CLS and SEP tokens
        input_ids.insert(0, self.cls_token_id)
        input_ids.append(self.sep_token_id)
        
        attention_mask.insert(0, 0)
        attention_mask.append(0)

        tags.insert(0, self.id_of_o) # tag of CLS
        tags.append(self.id_of_o) # tag of SEP

        tokens.insert(0, self.pad_token) # token for CLS
        tokens.append(self.pad_token) # token for SEP

        # cast to tensors
        input_ids =  torch.LongTensor(input_ids)
        attention_mask =  torch.LongTensor(attention_mask)
        tags = torch.LongTensor(tags)

        #print(input_ids.shape, attention_mask.shape, tags.shape)

        assert input_ids.shape == attention_mask.shape
        assert len(input_ids) == len(attention_mask)


        return input_ids, attention_mask, tags, tokens