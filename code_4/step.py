import torch
import torch.nn.functional as F
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import notebook

def training_step(model, optimizer, scheduler, data_loader, device):
    model.train()  # Set the model to train mode
    train_loss = {'sc':0, 'pc':0, 'cls': 0, 'loss':0}
    train_correct = 0
    train_total = 0

    all_labels = []
    all_predicted = []
    batch_idx = 0
    for batch in notebook.tqdm(data_loader):
        optimizer.zero_grad()  # Zero out gradients

        for key, tensor in batch.items():
            batch[key] = tensor.to(device)
        
        # Forward pass
        outputs = model(batch)

        # Calculate loss
        loss = outputs['loss']
        sc_loss = outputs['sc_loss'] * 10
        pc_loss = outputs['pc_loss'] * 10
        cls_loss = outputs['cls_loss'] / 10

        train_loss['loss'] = train_loss['loss'] + loss.detach().item()
        train_loss['sc'] = train_loss['sc'] + sc_loss.detach().item()
        train_loss['pc'] = train_loss['pc'] + pc_loss.detach().detach().item()
        train_loss['cls'] = train_loss['cls'] + cls_loss.detach().detach().item()

        if batch_idx % 50 == 0:
            print(f'After {batch_idx} steps: loss {loss} cls_loss {cls_loss}, pc_loss {pc_loss}, sc_loss {sc_loss}')

        # Backward pass and optimization
        loss = loss + pc_loss + sc_loss + cls_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_idx +=1

    scheduler.step()
    batch_idx = 0

    # Calculate epoch statistics
    train_loss['cls'] = train_loss['cls'] / len(data_loader)
    train_loss['pc'] = train_loss['pc'] / len(data_loader)
    train_loss['sc'] = train_loss['sc'] / len(data_loader)
    train_loss['loss'] = train_loss['loss'] / len(data_loader)
    return train_loss





def validation_step(model, data_loader, inv_label_map, device):
    model.eval()
    dev_loss = 0.0
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for batch in data_loader:

            for key, tensor in batch.items():
                batch[key] = tensor.to(device)
            
            outputs = model(batch, train=False)

            aligned_labels = outputs['aligned_labels']
            aligned_logits = outputs['aligned_logits']
            
            #print(aligned_logits.shape, aligned_labels.shape)
            
            dev_loss += F.cross_entropy(aligned_logits.view(-1,42), aligned_labels.view(-1)).item()
            
            
            aligned_labels = aligned_labels.detach().cpu().numpy()
            aligned_preds = np.argmax(aligned_logits.detach().cpu().numpy(), axis=-1)
            
            all_labels.extend(aligned_labels)
            all_predicted.extend(aligned_preds)


    all_labels =  [inv_label_map[label] for label in all_labels]
    all_predicted = [inv_label_map[label] for label in all_predicted]
    f1 = f1_score([all_labels], [all_predicted])
    dev_loss /= len(data_loader)
    
    return f1, dev_loss