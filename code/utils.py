import numpy as np
import torch
import random
import copy
import pandas as pd
import json
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

def load_npy(file_path):
    """
    Load a .npy file and return its content as a NumPy array.

    Parameters:
        file_path (str): The path to the .npy file.

    Returns:
        numpy.ndarray: The content of the .npy file as a NumPy array.
    """
    try:
        data = np.load(file_path)
        return data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except:
        print(f"An error occurred while loading '{file_path}'.")
        return None


def load_json(path):
    # Open the JSON file
    with open(path, 'r') as f:
        # Load JSON data from file
        data = json.load(f)
    print(f'data loaded from path {path}')
    return data

def print_classification_report(y_true, y_pred):
    """
    Compute and print the classification report for two label tensors.

    Parameters:
        y_true (torch.Tensor): The true label tensor of shape (1, 1338).
        y_pred (torch.Tensor): The predicted label tensor of shape (1, 1338).

    Returns:
        None
    """
    # Convert tensors to NumPy arrays
    y_true_np = y_true.numpy().flatten()
    y_pred_np = y_pred.numpy().flatten()
    

    # Compute classification report
    report = classification_report(y_true_np, y_pred_np, zero_division=0)

    # Print the classification report
    print(report)
    calculate_f1_scores(y_true, y_pred)

    
# Function to flatten and normalize the data including the tag values
def normalize_data(items):
    normalized_data = []
    for item in items:
        sentence_id = item['global_sentence_id']
        for token_info in item['tokens']:
            token = token_info['token']
            tv = token_info['tags'][0]
            values = tv['value']
            normalized_data.append({'sid': sentence_id, 'word': token, 'tag':values})
    return normalized_data


def labels_stats(df):
    tags = df.tag.unique()
    print(f'unique primary tags size {len(tags)}')

def load_task1_data():
    train_data = load_json('/content/drive/MyDrive/anlp24/split70.json')
    dev_data = load_json('/content/drive/MyDrive/anlp24/split10.json')
    
    # Normalize the data
    normalized_train = normalize_data(train_data)
    normalized_dev = normalize_data(dev_data)
    # Create a DataFrame
    train_df = pd.DataFrame(normalized_train)
    print(f'Train data normalized, columns {train_df.columns} with shape {train_df.shape}')
    labels_stats(train_df)
    dev_df = pd.DataFrame(normalized_dev)
    print(f'Dev data normalized, columns {dev_df.columns} with shape {dev_df.shape}')
    labels_stats(dev_df)
    
    return train_df, dev_df

def calculate_f1_scores(ground_truth, predictions):
    # Convert PyTorch tensors to NumPy arrays
    ground_truth_np = ground_truth.numpy().flatten()
    predictions_np = predictions.numpy().flatten()
    
    # Calculate and print F1-scores
    weighted_f1 = f1_score(ground_truth_np, predictions_np, average='weighted')*100
    micro_f1 = f1_score(ground_truth_np, predictions_np, average='micro')*100
    macro_f1 = f1_score(ground_truth_np, predictions_np, average='macro')*100
    print(f"Macro F1-score: {macro_f1:.2f}")
    print(f"Micro F1-score: {micro_f1:.2f}")
    print(f"Weighted F1-score: {weighted_f1:.2f}")
