import torch
from functools import partial
import re
import itertools
import datasets


class BertSeqTransform:
    def __init__(self, bert_model, tokenizer,  label_map, sub_labels_map, max_seq_len=512):
        self.tokenizer = tokenizer
        self.encoder = partial(
            self.tokenizer.encode,
            max_length=max_seq_len,
            truncation=True,
        )
        self.max_seq_len = max_seq_len
        self.label_map = label_map
        self.sub_labels_map = sub_labels_map
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
        
    def get_subtags_onehot_encoding(self, l2_tags, l3_tags):
        sub_tags = []
        if l2_tags:
            sub_tags += l2_tags
        if len(l3_tags) > 0:
            sub_tags += l3_tags
        
        one_hot = [torch.zeros(len(self.sub_labels_map))]
        for tag in sub_tags:
            one_hot[self.sub_labels_map[tag]] = 1
        return one_hot
        


    def __call__(self, segment):
        input_ids, tags, sub_tags, tokens = list(), list(), list(), list()

        for token in segment:
            # Sometimes the tokenizer fails to encode the word and return no input_ids, in that case, we use the input_id for [UNK]
            token_input_ids = self.encoder(token.text)[1:-1] or self.pad_token_id
            input_ids += token_input_ids
            
            # append tags with the tag of O(outside) if the word has mutliple tokens
            tags += [self.label_map[token.main_tag[0]]] #actual tag
            tags += [self.id_of_o] * (len(token_input_ids) - 1) #tag of O

            one_hot = self.get_subtags_onehot_encoding(token.l2_tags, token.l3_tags)
            sub_tags += one_hot
            sub_tags += [torch.zeros(len(self.sub_labels_map))] * (len(token_input_ids) - 1)

            # append tokens with UNK if the word has mutliple tokens
            tokens += [token] + [self.pad_token] * (len(token_input_ids) - 1)
        
        # Truncate to max_seq_len if needed
        truncat_length = (len(input_ids) + 2) - self.max_seq_len
        if truncat_length > 0:
            text = " ".join([t.text for t in tokens if t.text != "UNK"])
            print("Truncating the sequence %s to %d", text, self.max_seq_len - 2)
            input_ids = input_ids[:self.max_seq_len - 2]
            tags = tags[:self.max_seq_len - 2]
            sub_tags = sub_tags[:self.max_seq_len - 2]
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

        sub_tags.insert(0, torch.zeros(len(self.sub_labels_map))) # tag of CLS
        sub_tags.append(torch.zeros(len(self.sub_labels_map)))

        tokens.insert(0, self.pad_token) # token for CLS
        tokens.append(self.pad_token) # token for SEP

        # cast to tensors
        input_ids =  torch.LongTensor(input_ids)
        attention_mask =  torch.LongTensor(attention_mask)
        tags = torch.LongTensor(tags)
        print(sub_tags)
        sub_tags = torch.stack(sub_tags)

        #print(input_ids.shape, attention_mask.shape, tags.shape)

        assert input_ids.shape == attention_mask.shape
        assert len(input_ids) == len(attention_mask)
        assert len(tags)  == len(sub_tags)


        return input_ids, attention_mask, tags, sub_tags, tokens