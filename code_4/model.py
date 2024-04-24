import copy

from transformers import BertModel

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np



class ProtoSimModel(nn.Module):

    def __init__(self, rhetorical_role, embedding_width, device):
        nn.Module.__init__(self)
        self.prototypes = nn.Embedding(rhetorical_role, embedding_width)
        self.classification_layer = nn.Linear(embedding_width, rhetorical_role)
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.device=device

    def forward(self, entity_embedding, entity_id):
        #print('prototypes: ',self.prototypes)
        #print('entity_id: ', entity_id)
        protos = self.prototypes(entity_id)
        
        protos = F.normalize(protos, p=2, dim=-1)  # Normalize prototype embeddings
        entity_embedding = F.normalize(entity_embedding, p=2, dim=-1)  # Normalize input embeddings
        
        similarity = torch.sum(protos * entity_embedding, dim=-1)  # Cosine similarity
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

            #print('labele: ',label ,'break it to: ',label_embeddings.shape, other_embeddings.shape)

            p_sim, _ = self.forward(label_embeddings, label)
            n_sim, _ = self.forward(other_embeddings, label)
            
            if p_sim.shape[0]>0:
              cluster_loss += -torch.mean(torch.log(p_sim + 1e-5))
            if n_sim.shape[0]>0:
              cluster_loss += -torch.mean(torch.log(1 - n_sim + 1e-5))

            #cluster_loss += -(torch.mean(torch.log(p_sim + 1e-5)) + torch.mean(torch.log(1 - n_sim + 1e-5)))
            #print('cluster loss: ',cluster_loss)
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

            p_label = label.repeat(p_predicted_role.size(0)).type(torch.FloatTensor).to(self.device)
            
            if p_predicted_role.shape[0] > 0:
              cls_loss += self.cross_entropy(p_predicted_role, p_label)
            
            if n_predicted_role.shape[0] > 0:
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

            
            if p_sim.shape[0]>0:
              cluster_loss += -torch.mean(torch.log(p_sim + 1e-5))
            
            
            if len(n_sim_list)>0:
              n_sim = torch.mean(torch.stack(n_sim_list), dim=0)
              cluster_loss += -torch.mean(torch.log(1 - n_sim + 1e-5))

            #cluster_loss += -torch.mean(torch.log(p_sim + 1e-5)) #+ torch.mean(torch.log(1 - n_sim + 1e-5)))
            #print('cluster loss: ',cluster_loss)
        cluster_loss /= batch_size

        return cluster_loss




class BertTokenEmbedder(torch.nn.Module):
    def __init__(self, config):
        super(BertTokenEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_model"])
        self.bert_hidden_size = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask, labels):
        batch_size, tokens = input_ids.shape

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
       
        # shape (batch_size, tokens, 768)
        bert_embeddings = outputs.last_hidden_state

        return bert_embeddings

class BertHSLN(torch.nn.Module):
    '''
    Model for Baseline, Sequential Transfer Learning and Multitask-Learning with all layers shared (except output layer).
    '''
    def __init__(self, config, label_map):
        super(BertHSLN, self).__init__()
        self.num_labels = config['num_labels']
        self.label_map = config['label_map']
        self.inv_label_map = config['inv_label_map']
        self.device = config['device']

        self.bert = BertTokenEmbedder(config)

        # Jin et al. uses DROPOUT WITH EXPECTATION-LINEAR REGULARIZATION (see Ma et al. 2016),
        # we use instead default dropout
        self.dropout = torch.nn.Dropout(config["dropout"])
        
        self.hidden_size = self.bert.bert_hidden_size

        # Initialize ProtoSimModel
        self.proto_sim_model = ProtoSimModel(self.num_labels, self.hidden_size, self.device)

        self.classifier = torch.nn.Linear(self.hidden_size, self.num_labels)
        
    
    def align_logits(self, logits, label_ids):
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.detach().cpu().numpy()
        preds = np.argmax(logits, axis=-1)


        ignored_index = torch.nn.CrossEntropyLoss().ignore_index

        valid_mask = label_ids != ignored_index

        # Use boolean indexing to filter out the embeddings and logits
        filtered_preds = preds[valid_mask]
        filtered_labels = label_ids[valid_mask]
        filtered_logits = logits[valid_mask]

        # Map labels using inv_label_map, vectorized operation
        aligned_labels = [self.inv_label_map[label.item()] for label in filtered_labels]
        aligned_preds = [self.inv_label_map[label.item()] for label in filtered_preds]

        assert len(aligned_preds) == len(aligned_labels)
        
        return aligned_preds, aligned_labels, filtered_labels, filtered_logits

    def align(self, logits, embeddings, label_ids):
        
        logits = logits
        label_ids = label_ids
        embeddings = embeddings

        ignored_index = torch.nn.CrossEntropyLoss().ignore_index
        valid_mask = label_ids != ignored_index

        # Use boolean indexing to filter out the embeddings and logits
        filtered_labels = label_ids[valid_mask]
        filtered_logits = logits[valid_mask]
        filtered_embeddings = embeddings[valid_mask]
        
        return filtered_logits, filtered_embeddings, filtered_labels


    def forward(self, batch, labels=None, get_embeddings = False, train=True):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        batch_size, tokens = batch["input_ids"].shape
        

        # shape (batch_size, tokens, hidden_size)
        bert_embeddings = self.bert(input_ids, attention_mask, labels)
        bert_embeddings = self.dropout(bert_embeddings)
        #print('bert_embeddings:',bert_embeddings.shape)

        # shape (batch_size, tokens, num_label)
        logits = self.classifier(bert_embeddings)
        #print('logits:',logits.shape)

        # align data
        #print('align data')
        logits, embeddings, labels = self.align(logits, bert_embeddings, labels)
        #print('logits', logits.shape) 
        #print('embeddings', embeddings.shape) 
        #print('labels', labels.shape) 


        output = {}
        if train:
                loss = F.cross_entropy(logits.view(-1,42), labels.view(-1))
                #print('loss: ',loss)
                pc_loss = self.proto_sim_model.get_proto_centric_loss(embeddings, labels)
                #print('------pc------:',pc_loss)
                sc_loss = self.proto_sim_model.get_sample_centric_loss(embeddings, labels)
                #print('------sc------:',pc_loss)
                cls_loss = self.proto_sim_model.get_classification_loss(embeddings, labels)
                #print('------cls------:',pc_loss)
                
                output['loss'] = loss
                output['pc_loss'] = pc_loss
                output['sc_loss'] = sc_loss
                output['cls_loss'] = cls_loss
                #output['logits']=logits
        else:
                logits, _, labels = self.align(logits, embeddings, labels)
                output['aligned_labels'] = labels
                output['aligned_logits'] = logits


        if get_embeddings:
            return output, embeddings
        else:
            return output