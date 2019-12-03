import tensorflow as tf
from tqdm import tqdm


def get_data(window_size, test_fraction=0.1, shuffle=True):
    assert 0.01 <= test_fraction <= 0.99

    with open('vocab.txt', 'r', encoding='utf-8') as vocab_file:
        vocab_lines = map(lambda line: line.strip().split(), vocab_file)
        vocab = {}
        reverse_vocab = {}
        for vocab_line in vocab_lines:
            vocab[vocab_line[0]] = vocab_line[1]
            reverse_vocab[vocab_line[1]] = vocab_line[0]

    pad_id = len(vocab)
    vocab['*pad*'] = pad_id
    reverse_vocab[pad_id] = '*pad*'

    with open('updated_genres.txt', 'r',encoding='utf-8') as genres_file:
        num_genres = 0
        for _ in genres_file:
            num_genres += 1

    def fit_to_size(tokens):
        num_relevant_tokens = min(len(tokens), window_size)
        return tokens[:num_relevant_tokens] + [pad_id] * (window_size - num_relevant_tokens)

    def convert_labels_to_onehot(labels):
        out = [0.0] * num_genres
        for label in labels:
            out[label] = 1.0

        return tf.convert_to_tensor(out)

    with open('tokens.txt', 'r',encoding='utf-8') as tokens_file:
        lines = [line.strip() for line in tokens_file]
    
    num_datapoints = len(lines)

    def get_iterator(iterator, desc):
        return tqdm(iterator, desc=desc, leave=True, total=num_datapoints)

    tokenized_lines = map(lambda line: list(map(int, line.split(","))), lines)

    tokens_data = tf.convert_to_tensor(list(map(fit_to_size, get_iterator(tokenized_lines, "tokenizing and fitting"))))

    with open('labels_converted.txt', 'r',encoding='utf-8') as labels_file:
        labels = tf.convert_to_tensor(list(map(convert_labels_to_onehot, map(lambda line: map(int, line.strip().split()), get_iterator(labels_file, "reading and converting labels")))))

    indices = tf.range(tokens_data.shape[0])
    if shuffle:
        indices = tf.random.shuffle(indices)
    
    split_point = int((1 - test_fraction) * len(tokens_data))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    train_data = tf.gather(tokens_data, train_indices)
    train_labels = tf.gather(labels, train_indices)
    test_data = tf.gather(tokens_data, test_indices)
    test_labels = tf.gather(labels, test_indices)

    return train_data, train_labels, test_data, test_labels, len(vocab), num_genres, pad_id
