import copy

from transformers import BertModel

import torch.nn as nn
import torch.nn.functional as F
import torch


class ProtoSimModel(nn.Module):

    def __init__(self, rhetorical_role, embedding_width):
        nn.Module.__init__(self)
        self.prototypes = nn.Embedding(rhetorical_role, embedding_width)
        self.classification_layer = nn.Linear(embedding_width, rhetorical_role)
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, role_embedding, role_id):
        protos = self.prototypes(role_id)
        
        protos = F.normalize(protos, p=2, dim=-1)  # Normalize prototype embeddings
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

        for label in torch.unique(labels):
            label_mask = labels == label
            other_mask = labels != label


            label_embeddings = embeddings[label_mask]
            other_embeddings = embeddings[other_mask]
            other_labels = labels[other_mask]  # Capture the labels for other embeddings

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




class BertTokenEmbedder(torch.nn.Module):
    def __init__(self, config):
        super(BertTokenEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_model"])
        self.bert_hidden_size = self.bert.config.hidden_size

    def forward(self, batch):
        batch_size, sentences, tokens = batch["input_ids"].shape

        attention_mask = batch["attention_mask"].view(-1, tokens)
        input_ids = batch["input_ids"].view(-1, tokens)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
       
        # shape (documents*sentences, tokens, 768)
        bert_embeddings = outputs[0]

        return bert_embeddings

class BertHSLN(torch.nn.Module):
    '''
    Model for Baseline, Sequential Transfer Learning and Multitask-Learning with all layers shared (except output layer).
    '''
    def __init__(self, config, num_labels):
        super(BertHSLN, self).__init__()
        self.use_crf = config['use_crf']
        self.num_labels = num_labels
        self.bert = BertTokenEmbedder(config)

        # Jin et al. uses DROPOUT WITH EXPECTATION-LINEAR REGULARIZATION (see Ma et al. 2016),
        # we use instead default dropout
        self.dropout = torch.nn.Dropout(config["dropout"])
        
        self.lstm_hidden_size = config["word_lstm_hs"]

        # Initialize ProtoSimModel
        self.proto_sim_model = ProtoSimModel(self.num_labels, self.lstm_hidden_size * 2)

        self.classifier = torch.nn.Linear(self.lstm_hidden_size * 2, self.num_labels)
        
        
    def forward(self, batch, labels=None, get_embeddings = False):

        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')

        batch_size, sentences, tokens = batch["input_ids"].shape
        

        # shape (documents*sentences, tokens, 768)
        bert_embeddings = self.bert(input_ids, attention_mask, labels)

        # in Jin et al. only here dropout
        bert_embeddings = self.dropout(bert_embeddings)

        output = {}
        if labels is not None:
                logits = logits.squeeze()
                labels = labels.squeeze()
                predicted_labels = torch.argmax(logits, dim=1)
                output['predicted_label'] = predicted_labels

                loss = F.cross_entropy(logits, labels)
                pc_loss = self.proto_sim_model.get_proto_centric_loss(bert_embeddings, labels.unsqueeze(0))
                sc_loss = self.proto_sim_model.get_sample_centric_loss(bert_embeddings, labels.unsqueeze(0))
                cls_loss = self.proto_sim_model.get_classification_loss(bert_embeddings, labels.unsqueeze(0))
                
                output['loss'] = loss
                output['pc_loss'] = pc_loss
                output['sc_loss'] = sc_loss
                output['cls_loss'] = cls_loss
                
                output['logits']=logits
        else:
                logits = logits.squeeze()
                predicted_labels = torch.argmax(logits, dim=1)
                output['predicted_label'] = predicted_labels
                output['logits']=logits


        if get_embeddings:
            return output, bert_embeddings
        else:
            return output