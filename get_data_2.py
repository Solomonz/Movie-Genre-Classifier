import numpy as np
from tqdm import tqdm
import json

def get_data(window_size, test_fraction=0.15, shuffle=True):
    assert 0.01 <= test_fraction <= 0.99

    #TODO: Fix the json!

    vocab = dict()
    with open('vocab_2.txt', 'r') as file:
        vocab = json.loads(file.readline())
    genres = dict()
    with open('genres_2.txt', 'r') as file:
        genres = json.loads(file.readline())
    
    tokens = []
    labels = []
    with open('tokens_2.txt', 'r') as file:
        for line in file:
            splitted = [int(s) for s in line.split(',')]
            if len(splitted)>window_size:
                splitted = splitted[:window_size]
                        
            splitted = np.append(splitted,np.zeros(window_size-len(splitted)))
            # print(splitted)
            tokens.append(splitted)
    tokens = np.array(tokens, dtype="int32")
    
    with open('labels_2.txt', 'r') as file:
        for line in file:
            splitted = [int(s) for s in line.strip().split(',')]
            lab = np.sum(np.eye(len(genres))[splitted], axis=0)
            labels.append(lab)

    labels = np.array(labels, dtype="int32")
    
    assert tokens.shape[0] == labels.shape[0]

    indices = np.arange(tokens.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    


    split_point = int((1 - test_fraction) * len(tokens))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    train_data = np.take(tokens, train_indices, axis=0)
    train_labels = np.take(labels, train_indices, axis=0)
    test_data = np.take(tokens, test_indices, axis=0)
    test_labels = np.take(labels, test_indices, axis=0)
    return train_data, train_labels, test_data, test_labels, len(vocab)+1, len(genres), 0
