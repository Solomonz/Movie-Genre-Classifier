import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from get_data import get_binary_data
import numpy as np
import sys
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import f1_score


class Model(tf.keras.Model):
    def __init__(self, vocab_size, window_size):

        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = window_size
        self.embedding_size = 200
        self.hidden_size1 = 256
        self.rnn_size = 500
        self.batch_size = 128

        self.E = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.gru = tf.keras.layers.GRU(self.rnn_size, return_sequences=True, return_state=True)
        self.W1 = tf.keras.layers.Dense(self.hidden_size1, activation="relu")
        self.W2 = tf.keras.layers.Dense(2)
    
        self.epsilon = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.epsilon)


    def call(self, inputs):
        embeddings = self.E(inputs)
        _, output_state = self.gru(embeddings)

        hidden = self.W1(output_state)
        logits = self.W2(hidden)
        return logits

    def loss(self, logits, labels):
        """
        Calculates cross entropy loss of the prediction
        
        :param logits: a matrix of shape (batch_size, 2) as a tensor
        :param labels: onehot matrix of shape (batch_size, 2) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
        return tf.reduce_mean(loss)

    def accuracy(self, logits, labels):
        """
        Calculates accuracy of given predictions
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32)).numpy()
        

def train(model, train_inputs, train_labels):
    batch_id = 0

    for batch_start in tqdm(list(range(0, len(train_inputs), model.batch_size)), desc="training"):
        batch_end = min(batch_start + model.batch_size, len(train_inputs))
        inputs = train_inputs[batch_start:batch_end]
        labels = train_labels[batch_start:batch_end]
        
        with tf.GradientTape() as tape:
            logits = model.call(inputs)
            loss = model.loss(logits, tf.argmax(labels, 1))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if batch_id % 10 == 0:
            print(loss.numpy())

        batch_id += 1

def test(model, test_inputs, test_labels):
    acc = []

    for batch_start in tqdm(list(range(0, len(test_inputs), model.batch_size)), desc="testing"):
        inputs = test_inputs[batch_start:min(batch_start + model.batch_size, len(test_inputs))]
        labels = test_labels[batch_start:min(batch_start + model.batch_size, len(test_labels))]
        logits = model.call(inputs)
        acc.append(model.accuracy(logits,labels))
    print()

    return sum(acc) / len(acc)



def main():
    window_size = 80
    equiv_class_0 = {'Romance'}
    equiv_class_1 = {'Thriller'}
    train_inputs, train_labels, test_inputs, test_labels, vocab_len = get_binary_data(window_size, equiv_class_0, equiv_class_1)

    model = Model(vocab_len, window_size)
    epochs = 5
    
    for e in range(epochs):
        print("Accuracy: ", test(model, test_inputs, test_labels))
        train(model, train_inputs, train_labels)

    print("Accuracy: ", test(model, test_inputs, test_labels))


if __name__ == '__main__':
    main()
