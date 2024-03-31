import torch
import pytorch_lightning as pl
from transformers import BertModel, BertConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchmetrics
from seqeval.metrics import f1_score, classification_report

class LightningBertNer(pl.LightningModule):
    def __init__(self, args):
        super(LightningBertNer, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.bert_config = BertConfig.from_pretrained(args.bert_dir)
        hidden_size = self.bert_config.hidden_size
        self.lstm_hidden = 128
        self.max_seq_len = args.max_seq_len
        self.bilstm = torch.nn.LSTM(hidden_size, self.lstm_hidden, 1, bidirectional=True, batch_first=True, dropout=0.1)
        self.linear = torch.nn.Linear(self.lstm_hidden * 2, args.num_labels)
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_labels)
        #self.val_f1 = torchmetrics.F1Score(task="multiclass",num_classes=args.num_labels, average='macro')
        self.all_labels = args.all_labels
        self.inv_label_map = {i: label for i, label in enumerate(self.all_labels)}


    def forward(self, input_ids, attention_mask, labels=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = bert_output[0]  # [batchsize, max_len, 768]
        batch_size = seq_out.size(0)
        seq_out, _ = self.bilstm(seq_out)
        seq_out = seq_out.contiguous().view(-1, self.lstm_hidden * 2)
        seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1)
        logits = self.linear(seq_out)
        outputs = {'logits': logits, 'labels': labels}
        if labels is not None:
            loss = self.cross_entropy(logits.view(-1, self.args.num_labels), labels.view(-1))
            outputs['loss'] = loss
        return outputs
    
    def training_step(self, batch, batch_idx):
        #input_ids, attention_mask,token_type_ids, labels = batch
        input_ids, attention_mask, token_type_ids, labels = batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['label']
        outputs = self.forward(input_ids, attention_mask, labels)
        loss = outputs['loss']
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['label']
        outputs = self.forward(input_ids, attention_mask, labels)
        val_loss = outputs['loss']
        logits = outputs['logits']
        
        # Calculate predictions
        preds_list, out_label_list = self.align_predictions(logits, labels)
        preds = torch.argmax(logits, dim=2)
        labels = labels.view(-1)
        preds = preds.view(-1)

        # Update metrics
        val_accuracy_value = self.val_accuracy(preds, labels)
        val_f1_value = f1_score(out_label_list, preds_list)


        #print(pres.shape, labels.shape)

        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)    
        self.log('val_accuracy', val_accuracy_value, on_step=True, on_epoch=True, prog_bar=True, logger=True)    
        self.log('val_f1_value', val_f1_value, on_step=True, on_epoch=True, prog_bar=True, logger=True)    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


    def align_predictions(self, predictions, label_ids):
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(self.inv_label_map[label_ids[i][j]])
                    preds_list[i].append(self.inv_label_map[preds[i][j]])

        return preds_list, out_label_list
