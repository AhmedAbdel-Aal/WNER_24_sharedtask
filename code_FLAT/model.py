import torch
import torch.nn as nn
from transformers import BertModel

class BertWithMLPs(nn.Module):
    def __init__(self, bert_model='aubmindlab/bert-base-arabertv2', num_main_labels=43, num_subtype_labels=63):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(0.1)
        
        # MLP for main labels
        self.main_label_classifier = nn.Linear(self.bert.config.hidden_size, num_main_labels)
        
        # MLP for subtype labels
        self.subtype_label_classifier = nn.Linear(self.bert.config.hidden_size, num_subtype_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Predict main labels
        main_labels_logits = self.main_label_classifier(pooled_output)
        
        # Predict subtype labels
        subtype_labels_logits = self.subtype_label_classifier(pooled_output)
        
        return main_labels_logits, subtype_labels_logits

# Initialize model
model = BertWithMLPs()
