import numpy as np
import torch
import random
import copy
import pandas as pd
import json

def load_json(path):
    # Open the JSON file
    with open(path, 'r') as f:
        # Load JSON data from file
        data = json.load(f)
    print(f'data loaded from path {path}')
    return data


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
    train_data = load_json('../data/subtask1/split70.json')
    dev_data = load_json('../data/subtask1/split10.json')
    
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