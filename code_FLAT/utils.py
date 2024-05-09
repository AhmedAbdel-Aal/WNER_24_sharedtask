import numpy as np
import json

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
