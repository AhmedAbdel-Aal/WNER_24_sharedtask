import numpy as np
import torch


def choose(embeddings, label_ids, logits, inv_label_map):
    label_ids = label_ids.detach().cpu().numpy()
    #logits = torch.softmax(logits, dim=-1)
    logits = logits.detach().cpu().numpy()
    embeddings = embeddings.detach().cpu().numpy()

    ignored_index = torch.nn.CrossEntropyLoss().ignore_index

    valid_mask = label_ids != ignored_index

    # Use boolean indexing to filter out the embeddings and logits
    filtered_embeddings = embeddings[valid_mask]
    filtered_logits = logits[valid_mask]
    filtered_labels = label_ids[valid_mask]

    # Map labels using inv_label_map, vectorized operation
    out_label_list = [inv_label_map[label.item()] for label in filtered_labels]

    assert len(filtered_embeddings) == len(filtered_logits)
    assert len(filtered_embeddings) == len(filtered_labels)
    return torch.tensor(filtered_embeddings), out_label_list, torch.tensor(filtered_logits)


def postprocess_logits_to_labels(embedding_test, keys, datastore_values_mapped, K):
  #i = 2
  batch = embedding_test.shape[0]
  num_labels=43
  hidden_size = embedding_test.shape[-1]
  embedding_test = embedding_test.view(-1, hidden_size)

  # cosine similarity
  sim = torch.mm(embedding_test, keys)
  norm_keys = torch.norm(keys, dim=0, keepdim=True)
  norm_embeddings = torch.norm(embedding_test, dim=1, keepdim=True)
  scores = sim / (norm_keys + 1e-10) / (norm_embeddings + 1e-10)

  # negative euclidean distance distance (p=2)
  #dists = torch.cdist(embedding_test, keys.transpose(0, 1))
  #scores = 1/(1+dists)


  scores = scores.view(1, embedding_test.shape[0], -1)


  topk_scores, topk_idxs = torch.topk(scores, dim=-1, k=K)

  datastore_size = keys.shape[1]
  knn_labels = datastore_values_mapped.unsqueeze(0)
  knn_labels = knn_labels.view(1, 1, datastore_size).expand(1, batch, datastore_size)
  knn_labels = knn_labels.gather(dim=-1, index=topk_idxs)

  sim_probs = torch.softmax(topk_scores/100, dim=-1)
  #print(sim_probs)
  knn_probabilities = torch.zeros_like(sim_probs[:, :, 0]).unsqueeze(-1).repeat([1,1,num_labels])
  knn_probabilities = knn_probabilities.scatter_add(dim=2, index=knn_labels, src=sim_probs)
  knn_probabilities = knn_probabilities.squeeze(0)
  #print(knn_probabilities)

  return knn_probabilities