import torch
import pytorch_lightning as pl
from transformers import BertModel, BertConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchmetrics
from seqeval.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

class ProtoSimModel(nn.Module):

    def __init__(self, rhetorical_role, embedding_width, device):
        nn.Module.__init__(self)
        self.rhetorical_role = rhetorical_role
        self.prototypes = nn.Embedding(rhetorical_role, embedding_width)
        self.classification_layer = nn.Linear(embedding_width, rhetorical_role)
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.device = device

    def forward(self, role_embedding, role_id):
        #print(role_embedding.shape)
        #print(role_id)
        if (role_id == -100).any().item():
          role_id = torch.tensor(self.rhetorical_role-1, device = self.device)
        #print(role_id)
        protos = self.prototypes(role_id)
        #print(protos)
        protos = F.normalize(protos, p=2, dim=-1)  # Normalize prototype embeddings
        #print(protos)
        role_embedding = F.normalize(role_embedding, p=2, dim=-1)  # Normalize input embeddings
        
        similarity = torch.sum(protos * role_embedding, dim=-1)  # Cosine similarity
        similarity = torch.exp(similarity)
        dist = 1-(1 / (1 + similarity))  # Cosine distance
        
        predict_role = self.classification_layer(protos)
        
        return dist, predict_role

    def get_proto_centric_loss(self, embeddings, labels):
        """
        prototypes centric view
        """
        batch_size = embeddings.size(1)
        cluster_loss = 0.0
        #print('labels: ', labels)
        #print('unique_labels', torch.unique(labels))
        for label in torch.unique(labels):
            label_mask = labels == label
            other_mask = labels != label
            #print(label_mask)

            label_embeddings = embeddings[label_mask]
            other_embeddings = embeddings[other_mask]
            other_labels = labels[other_mask]  # Capture the labels for other embeddings
            #print(label_embeddings.shape)
            #print(label, label.shape)
            p_sim, _ = self.forward(label_embeddings, label)
            n_sim, _ = self.forward(other_embeddings, label)

            cluster_loss += -(torch.mean(torch.log(p_sim + 1e-5)) + torch.mean(torch.log(1 - n_sim + 1e-5)))

        cluster_loss /= batch_size

        return cluster_loss


    def get_classification_loss(self, embeddings, labels):
        batch_size = embeddings.size(1)
        cls_loss = 0.0

        for label in torch.unique(labels):
            label_mask = labels == label
            other_mask = labels != label

            label_embeddings = embeddings[label_mask]
            other_embeddings = embeddings[other_mask]
            other_labels = labels[other_mask]

            _, p_predicted_role = self.forward(label_embeddings, label)
            _, n_predicted_role = self.forward(other_embeddings, other_labels)

            p_label = label.repeat(p_predicted_role.size(0)).type(torch.FloatTensor).cuda()

            cls_loss += self.cross_entropy(p_predicted_role, p_label)
            cls_loss += self.cross_entropy(n_predicted_role, other_labels)

        cls_loss /= batch_size
        return cls_loss
    
    def get_sample_centric_loss(self, embeddings, labels):
        """
        sample centric view
        """
        batch_size = embeddings.size(1)
        cluster_loss = 0.0

        unique_labels = torch.unique(labels)

        for label in unique_labels:
            label_mask = labels == label
            label_embeddings = embeddings[label_mask]

            # Calculate psim: distance between embeddings and their corresponding prototype
            p_sim, _ = self.forward(label_embeddings, label)

            # Calculate nsim: distance between embeddings and prototypes of different classes
            other_labels = unique_labels[unique_labels != label]
            n_sim_list = []
            for other_label in other_labels:
                n_sim, _ = self.forward(label_embeddings, other_label)
                n_sim_list.append(n_sim)

            n_sim = torch.mean(torch.stack(n_sim_list), dim=0)

            cluster_loss += -(torch.mean(torch.log(p_sim + 1e-5)) + torch.mean(torch.log(1 - n_sim + 1e-5)))

        cluster_loss /= batch_size

        return cluster_loss

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
        self.dropout = torch.nn.Dropout(args.dropout)
        self.proto_sim_model = ProtoSimModel(args.num_labels+1, self.lstm_hidden*2, args.device)
        
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_labels)
        self.all_labels = args.all_labels
        self.inv_label_map = {i: label for i, label in enumerate(self.all_labels)}


    def forward(self, input_ids, attention_mask, labels=None, get_embeddings=False):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = bert_output[0]  # [batchsize, max_len, 768]
        batch_size = seq_out.size(0)
        embeddings, _ = self.bilstm(seq_out)
        embeddings = self.dropout(embeddings)
        embeddings = embeddings.contiguous().view(-1, self.lstm_hidden * 2)
        seq_out = embeddings.contiguous().view(batch_size, self.max_seq_len, -1)
        logits = self.linear(seq_out)
        outputs = {'logits': logits, 'labels': labels}
        if labels is not None:
            loss = self.cross_entropy(logits.view(-1, self.args.num_labels), labels.view(-1))
            outputs['loss'] = loss
            #print(embeddings.shape)
            pc_loss = self.proto_sim_model.get_proto_centric_loss(embeddings, labels.view(-1))
            sc_loss = self.proto_sim_model.get_sample_centric_loss(embeddings, labels.view(-1))
            outputs['pc_loss'] = pc_loss
            outputs['sc_loss'] = sc_loss
        if get_embeddings:
            outputs['embeddings'] = embeddings
        return outputs
    

    def training_step(self, batch, batch_idx):
        #input_ids, attention_mask,token_type_ids, labels = batch
        input_ids, attention_mask, token_type_ids, labels = batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['label']
        outputs = self.forward(input_ids, attention_mask, labels)
        loss = outputs['loss']
        pc_loss = 5.0 * outputs['pc_loss'] 
        sc_loss = 5.0 * outputs['sc_loss'] 
        self.log('train_loss', loss)
        self.log('pc_loss', pc_loss)
        self.log('sc_loss', sc_loss)
        return loss + pc_loss + sc_loss
    
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
        val_accuracy_value = accuracy_score(out_label_list, preds_list)
        val_f1_value = f1_score(out_label_list, preds_list)
        val_precision = precision_score(out_label_list, preds_list)
        val_recall = recall_score(out_label_list, preds_list)

        self.log('val_loss', val_loss, on_step=True, prog_bar=True, logger=True)    
        self.log('val_accuracy', val_accuracy_value, on_epoch=True, prog_bar=True, logger=True)    
        self.log('val_f1_value', val_f1_value, on_epoch=True, prog_bar=True, logger=True)    
        self.log('val_precision', val_precision, on_epoch=True, prog_bar=True, logger=True)    
        self.log('val_recall', val_recall, on_epoch=True, prog_bar=True, logger=True)    
        return outputs

    def test_step(self, batch, batch_idx):
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
        val_accuracy_value = accuracy_score(out_label_list, preds_list)
        val_f1_value = f1_score(out_label_list, preds_list)
        val_precision = precision_score(out_label_list, preds_list)
        val_recall = recall_score(out_label_list, preds_list)

        self.log('val_loss', val_loss, on_step=True, prog_bar=True, logger=True)    
        self.log('val_accuracy', val_accuracy_value, on_epoch=True, prog_bar=True, logger=True)    
        self.log('val_f1_value', val_f1_value, on_epoch=True, prog_bar=True, logger=True)    
        self.log('val_precision', val_precision, on_epoch=True, prog_bar=True, logger=True)    
        self.log('val_recall', val_recall, on_epoch=True, prog_bar=True, logger=True)    
        return outputs



    def test_step_protos(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['label']
        outputs = self.forward(input_ids, attention_mask, labels, get_embeddings=True)
        return outputs['embeddings'], outputs['logits']




    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5)
        #scheduler = ExponentialLR(optimizer, gamma=0.99)  # gamma is the decay rate
        return {"optimizer": optimizer}#, "lr_scheduler": scheduler}


    def align_predictions(self, predictions, label_ids):
        label_ids = label_ids.detach().cpu().numpy()
        preds = torch.argmax(predictions, axis=2)
        preds = preds.detach().cpu().numpy()

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(self.inv_label_map[label_ids[i][j]])
                    preds_list[i].append(self.inv_label_map[preds[i][j]])

        return preds_list, out_label_list
