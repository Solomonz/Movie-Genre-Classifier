from get_data import get_data
import numpy as np
import sys
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import f1_score


class Model(tf.keras.Model):
    def __init__(self, vocab_size, num_genres, window_size, pad_id):

        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.num_genres = num_genres
        self.window_size = window_size
        self.pad_id = pad_id
        self.embedding_size = 400
        self.hidden_size1 = 256
        self.hidden_size2 = 128
        self.rnn_size = 256
        self.batch_size = 128

        self.epsilon = 0.015
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.epsilon)

        self.E = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.gru = tf.keras.layers.GRU(self.rnn_size, return_sequences=True, return_state=True)
        self.W1 = tf.keras.layers.Dense(self.hidden_size1, activation="relu")
        # self.W2 = tf.keras.layers.Dense(self.hidden_size2, activation="relu")
        self.W3 = tf.keras.layers.Dense(self.num_genres, activation='sigmoid')

    def call(self, inputs):
        embeddings = self.E(inputs)
        _, output_state = self.gru(embeddings)

        hidden1 = self.W1(output_state)
        # hidden2 = self.W2(hidden1)
        logits = self.W3(hidden1)

        return logits

    def loss(self, logits, labels):
        """
        Calculates cross entropy loss of the prediction
        
        :param logits: a matrix of shape (batch_size, num_genres) as a tensor
        :param labels: multihot matrix of shape (batch_size, num_genres) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)
        return tf.reduce_mean(loss)

    def accuracy(self, logits, labels):
        """
        Calculates accuracy of given predictions

        """
        pred = tf.dtypes.cast(tf.math.greater(logits, tf.constant(.5, shape=logits.shape)), tf.int32)
        return f1_score(labels,pred, average='micro')
        

def train(model, train_inputs, train_labels):
    indices = tf.random.shuffle(tf.range(train_inputs.shape[0]))
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)

    for batch_start in tqdm(list(range(0, len(train_inputs), model.batch_size)), desc="training"):
        inputs = train_inputs[batch_start:min(batch_start + model.batch_size, len(train_inputs))]
        labels = train_labels[batch_start:min(batch_start + model.batch_size, len(train_labels))]

        with tf.GradientTape() as tape:
            logits = model.call(inputs)
            loss = model.loss(logits, labels)

        if batch_start % 1280 == 0:
            print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
    indices = tf.random.shuffle(tf.range(test_inputs.shape[0]))
    test_inputs = tf.gather(test_inputs, indices)
    test_labels = tf.gather(test_labels, indices)

    acc = []

    for batch_start in tqdm(list(range(0, len(test_inputs), model.batch_size)), desc="testing"):
        inputs = test_inputs[batch_start:min(batch_start + model.batch_size, len(test_inputs))]
        labels = test_labels[batch_start:min(batch_start + model.batch_size, len(test_labels))]
        logits = model.call(inputs)
        acc.append(model.accuracy(logits,labels))

    return sum(acc)/len(acc)



def main():
    window_size = 60
    train_inputs, train_labels, test_inputs, test_labels, vocab_len, num_genres, pad_id = get_data(window_size=window_size)


    model = Model(vocab_len, num_genres, window_size, pad_id)
    epochs = 5
    
    for e in range(epochs):
        print("Accuracy: ", test(model, test_inputs, test_labels))
        train(model, train_inputs, train_labels)


if __name__ == '__main__':
    main()