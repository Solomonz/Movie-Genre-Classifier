import tensorflow as tf
import numpy as np
from get_data import get_data
import sys


class Model(tf.keras.Model):
    def __init__(self, vocab_size, num_genres):

        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.num_genres = num_genres
        self.window_size = 60
        self.embedding_size = 300
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.rnn_size = 256
        self.batch_size = 128

        self.epsilon = 0.015
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.epsilon)

        self.E = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.gru = tf.keras.layers.GRU(self.rnn_size, return_sequences=True, return_state=True)
        self.W1 = tf.keras.layers.Dense(self.hidden_size1, activation="relu")
        self.W2 = tf.keras.layers.Dense(self.hidden_size2, activation="relu")
        self.W3 = tf.keras.layers.Dense(self.num_genres, activation="softmax")

    def call(self, inputs):
        embeddings = self.E(inputs)
        _, output_state = self.gru(embeddings)

        hidden1 = self.W1(output_state)
        hidden2 = self.W2(hidden1)
        probs = self.W3(hidden2)

        return probs

    def loss(self, probs, labels):
        """
        Calculates cross entropy loss of the prediction
        
        :param logits: a matrix of shape (batch_size, num_genres) as a tensor
        :param labels: matrix of shape (batch_size, 1) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        return tf.nn.softmax_cross_entropy_with_logits(labels, logits)


def train(model, train_inputs, train_labels):
    indices = tf.random.shuffle(tf.range(train_inputs.shape[0]))
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)

    for batch_start in list(range(0, len(train_inputs), model.batch_size)):
        inputs = train_inputs[batch_start:min(batch_start + model.batch_size, len(train_inputs))]
        labels = train_labels[batch_start:min(batch_start + model.batch_size, len(train_labels))]

        with tf.GradientTape() as tape:
            logits, _ = model.call(inputs, None)
            loss = model.loss(logits, labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples
    
    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set

    Note: perplexity is exp(total_loss/number of predictions)

    """
    avg_CEL = tf.reduce_mean(model.loss(model.call(test_inputs, None)[0], tf.reshape(test_labels, (-1, model.window_size))))
    return tf.exp(avg_CEL)


def generate_sentence(word1, length, vocab,model):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    This is only for your own exploration. What do the sequences your RNN generates look like?
    
    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    reverse_vocab = {idx:word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits,previous_state = model.call(next_input,previous_state)
        out_index = np.argmax(np.array(logits[0][0]))

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))



def main():
    train_inputs, train_labels, test_inputs, test_labels, vocab_dict, num_genres = get_data()

    model = Model(len(vocab_dict), num_genres)

    #train_tokens = train_tokens[:len(train_tokens) - ((len(train_tokens) - 1) % (model.window_size * model.batch_size))]
    #test_tokens = test_tokens[:len(test_tokens) - ((len(test_tokens) - 1) % (model.window_size * model.batch_size))]

    #train_inputs = tf.reshape(tf.convert_to_tensor(train_tokens[:-1]), (-1, model.window_size))
    #train_labels = tf.reshape(tf.convert_to_tensor(train_tokens[1:]), (-1, model.window_size))

    #test_inputs = tf.reshape(tf.convert_to_tensor(test_tokens[:-1]), (-1, model.window_size))
    #test_labels = tf.reshape(tf.convert_to_tensor(test_tokens[1:]), (-1, model.window_size))

    train(model, train_inputs, train_labels)

    print(test(model, test_inputs, test_labels).numpy())

    
if __name__ == '__main__':
    main()
