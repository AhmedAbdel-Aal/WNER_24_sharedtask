import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transforms import BertSeqTransform



class Token:
    def __init__(self, text=None, pred_tag=None, gold_tag=None):
        """
        Token object to hold token attributes
        :param text: str
        :param pred_tag: str
        :param gold_tag: str
        """
        self.text = text
        self.gold_tag = gold_tag
        self.pred_tag = pred_tag

    def __str__(self):
        """
        Token text representation
        :return: str
        """
        
        gold_tags = "|".join(self.gold_tag)

        if self.pred_tag:
            pred_tags = "|".join([pred_tag["tag"] for pred_tag in self.pred_tag])
        else:
            pred_tags = ""

        if self.gold_tag:
            r = f"{self.text}\t{gold_tags}\t{pred_tags}"
        else:
            r = f"{self.text}\t{pred_tags}"

        return r


class NERDataset(Dataset):
    def __init__(
        self,
        examples=None,
        label_map=None,
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
        self.transform = BertSeqTransform(bert_model, label_map, max_seq_len=max_seq_len)
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
        input_ids, attention_mask, tags, tokens = zip(*batch)

        # Pad sequences in this batch
        # input_ids and attention_mask are padded with zeros
        # tags are padding with the index of the O tag
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        tags = pad_sequence(
            tags, batch_first=True, padding_value=self.label_map["O"]
        )
        return input_ids, attention_mask, tags, tokens
