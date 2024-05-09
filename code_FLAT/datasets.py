import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from transforms import BertSeqTransform




class Token:
    def __init__(self, text=None, main_tag=None, l2_tags=None, l3_tags=[]):
        """
        Token object to hold token attributes
        :param text: str
        :param pred_tag: str
        :param gold_tag: str
        """
        self.text = text
        self.main_tag = main_tag
        self.l2_tags = l2_tags
        self.l3_tags = l3_tags

    def __str__(self):
        """
        Token text representation
        :return: str
        """
        r = f"{self.text}\t{self.main_tag}\t{self.l2_tags}\t{self.l3_tags}"

        return r


class NERDataset(Dataset):
    def __init__(
        self,
        examples=None,
        label_map=None,
        sub_label_map=None,
        bert_model="aubmindlab/bert-base-arabertv2",
        max_seq_len=512,
    ):
        """
        The dataset that used to transform the segments into training data
        :param examples: list[[tuple]] - [[(token, tag), (token, tag), ...], [(token, tag), ...]]
                         You can get generate examples from -- arabiner.data.dataset.parse_conll_files
        :param vocab: vocab object containing indexed tags and tokens
        :param bert_model: str - BERT model
        :param: int - maximum sequence length
        """
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.transform = BertSeqTransform(bert_model, self.tokenizer, label_map, sub_label_map, max_seq_len=max_seq_len)
        self.examples = examples
        self.label_map = label_map

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        #print('get item')
        return self.transform(self.examples[item])


    def collate_fn(self, batch):
        """
        Collate function that is called when the batch is called by the trainer
        :param batch: Dataloader batch
        :return: Same output as the __getitem__ function
        """
        input_ids, attention_mask, tags, sub_tags, tokens = zip(*batch)

        # Pad sequences in this batch
        # input_ids and attention_mask are padded with zeros
        # tags are padding with the index of the O tag
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        tags = pad_sequence(
            tags, batch_first=True, padding_value=torch.nn.CrossEntropyLoss().ignore_index
        )
        sub_tags = pad_sequence(sub_tags, batch_first=True, padding_value=torch.nn.CrossEntropyLoss().ignore_index)

        r = {
            'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':tags,
            'sub_tags':sub_tags, 'tokens':tokens
             
             }
        return r
    
    def collate_fn_2(self, batch):
        """
        Collate function that is called when the batch is called by the trainer
        :param batch: Dataloader batch
        :return: Same output as the __getitem__ function
        """
        input_ids, attention_mask, tags, tokens = zip(*batch)

        # Pad sequences in this batch
        # input_ids and attention_mask are padded with zeros
        # tags are padding with the index of the O tag
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        tags = pad_sequence(
            tags, batch_first=True, padding_value=torch.nn.CrossEntropyLoss().ignore_index
        )
        r = {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':tags, 'tokens':tokens}
        return r
