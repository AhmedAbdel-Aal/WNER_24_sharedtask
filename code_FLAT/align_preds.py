import numpy as np
import torch
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def align_predictions(main_logits, main_labels, inv_main_label_map):
    main_logits = main_logits.detach().cpu().numpy()
    main_labels = main_labels.detach().cpu().numpy()
    
    preds = np.argmax(main_logits, axis=2)

    ignored_index = torch.nn.CrossEntropyLoss().ignore_index

    valid_mask = main_labels != ignored_index
    #print(valid_mask.shape)

    # Use boolean indexing to filter out the embeddings and logits
    filtered_preds = preds[valid_mask]
    filtered_labels = main_labels[valid_mask]
    #print(filtered_preds.shape, filtered_labels.shape)

    # Map labels using inv_label_map, vectorized operation
    preds_list = [inv_main_label_map[label.item()] for label in filtered_preds]
    out_label_list = [inv_main_label_map[label.item()] for label in filtered_labels]

    assert len(preds_list) == len(out_label_list)
    return [preds_list], [out_label_list]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def align_sub_logits(sub_logits, sub_labels, main_labels, inv_subtype_label_map):
    main_labels = main_labels.detach().cpu().numpy()
    
    ignored_index = torch.nn.CrossEntropyLoss().ignore_index
    valid_mask = main_labels != ignored_index

    # Use boolean indexing to filter out the embeddings and logits
    filtered_logits = sub_logits[valid_mask]
    filtered_labels = sub_labels[valid_mask]
    return filtered_logits, filtered_labels



def align_sub_predictions(sub_logits, sub_labels, main_labels, inv_subtype_label_map):
    sub_logits = sub_logits.detach().cpu().numpy()
    sub_labels = sub_labels.detach().cpu().numpy()
    main_labels = main_labels.detach().cpu().numpy()
    
    #print(sub_logits.shape)
    preds = sigmoid(sub_logits)
    #print(sub_logits.shape)

    ignored_index = torch.nn.CrossEntropyLoss().ignore_index
    valid_mask = main_labels != ignored_index

    # Use boolean indexing to filter out the embeddings and logits
    filtered_preds = preds[valid_mask]
    predicted_labels = (filtered_preds > 0.5).astype(int)

    filtered_labels = sub_labels[valid_mask]

    #print(filtered_preds.shape, filtered_labels.shape)

    # Map labels using inv_label_map, vectorized operation
    preds_list = []
    for token_labels in predicted_labels:
      token_subtags = []
      for idx, j in enumerate(token_labels):
        if j == 1:
          token_subtags.append(inv_subtype_label_map[idx])
      preds_list.append(token_subtags)
    
    out_label_list = []
    for token_labels in filtered_labels:
      token_subtags = []
      for idx, j in enumerate(token_labels):
        if j == 1:
          token_subtags.append(inv_subtype_label_map[idx])
      out_label_list.append(token_subtags)

    #preds_list = [inv_main_label_map[label.item()] for label in filtered_preds for k in label if label==1]
    #out_label_list = [inv_main_label_map[label.item()] for label in filtered_labels if label==1]
    p = []
    l = []
    for i,j in zip(preds_list, out_label_list):
      if len(i) > len(j):
          for k in range(len(j),len(i),1):
            j.append('O')
      elif len(i) < len(j):
          for k in range(len(i),len(j),1):
            i.append('O')
      p.extend(i)
      l.append(j)
    assert len(preds_list) == len(out_label_list)
    return preds_list, out_label_list

def compute_metrics(out_label_list, preds_list):
    return {
       # "accuracy_score": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }


def compute_metrics_subtypes(out_label_list, preds_list, subtype_label_map):
    out = ['O' for i in range(63)]
    for i in out_label_list:
          out[subtype_label_map['i']] = i

    preds = ['O' for i in range(63)]
    for i in preds_list:
          out[subtype_label_map['i']] = i

    return {
       # "accuracy_score": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out, preds),
        "recall": recall_score(out, preds),
        "f1": f1_score(out, preds),
    }