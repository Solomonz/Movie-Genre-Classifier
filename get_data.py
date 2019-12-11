import numpy as np
from tqdm import tqdm
from json import load, dump

def get_data(window_size, test_fraction=0.15, shuffle=True):
    assert 0.01 <= test_fraction <= 0.99

    vocab = dict()
    with open('processed/vocab.json', 'r') as vocab_file:
        vocab = load(vocab_file)

    genres = dict()
    with open('processed/genres.json', 'r') as genre_file:
        genres = load(genre_file)
    
    tokens = []
    labels = []
    with open('processed/tokens.txt', 'r') as tokens_file:
        for line in tokens_file:
            splitted = [int(s) for s in line.strip().split(',')]
            if len(splitted)>window_size:
                splitted = splitted[:window_size]
                        
            splitted = np.append(splitted,np.zeros(window_size-len(splitted)))
            # print(splitted)
            tokens.append(splitted)
    tokens = np.array(tokens, dtype="int32")
    
    with open('processed/labels.txt', 'r') as labels_file:
        for line in labels_file:
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
    return train_data, train_labels, test_data, test_labels, len(vocab), len(genres)


def get_binary_data(window_size, equivalence_class_0, equivalence_class_1, test_fraction=0.15, shuffle=True):
    assert 0.01 <= test_fraction <= 0.99

    vocab = dict()
    with open('processed/vocab.json', 'r') as vocab_file:
        vocab = load(vocab_file)

    genres = dict()
    with open('processed/genres.json', 'r') as genre_file:
        genres = load(genre_file)

    ec0 = set(map(genres.get, equivalence_class_0))
    ec1 = set(map(genres.get, equivalence_class_1))
    
    tokens = []
    labels = []
    with open('processed/tokens.txt', 'r') as tokens_file, open('processed/labels.txt', 'r') as labels_file:
        for tokens_line, labels_line in zip(tokens_file, labels_file):
            current_labels = set(labels_line.strip().split(','))
            overlaps = {len(current_labels & ec0), len(current_labels & ec1)}
            if 0 not in overlaps or len(overlaps) == 1:
                continue
            splitted = [int(s) for s in tokens_line.strip().split(',')]
            if len(splitted) > window_size:
                splitted = splitted[:window_size]
                        
            splitted.extend([vocab['*pad*']] *(window_size - len(splitted)))
            tokens.append(splitted)
            labels.append([len(current_labels & ec0) > 0, len(current_labels & ec1) > 0])

    tokens = np.array(tokens, dtype="int32")
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
    return train_data, train_labels, test_data, test_labels, len(vocab)
