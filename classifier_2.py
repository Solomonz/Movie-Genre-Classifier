from get_data_2 import get_data
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
        self.embedding_size = 300
        self.hidden_size1 = 256
        self.hidden_size2 = 128
        self.rnn_size = 500
        self.batch_size = 250

        self.E = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.gru = tf.keras.layers.GRU(self.rnn_size, return_sequences=True, return_state=True)
        self.W1 = tf.keras.layers.Dense(self.hidden_size1, activation="relu")
        # self.W2 = tf.keras.layers.Dense(self.hidden_size2, activation="relu")
        self.W3 = tf.keras.layers.Dense(self.num_genres, activation='sigmoid')
    
        self.epsilon = 0.015
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.epsilon)


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
        loss = tf.keras.losses.binary_crossentropy(labels, logits,from_logits=False)
        return tf.reduce_mean(loss)

    def accuracy(self, logits, labels):
        """
        Calculates accuracy of given predictions

        """
        pred = tf.dtypes.cast(tf.math.greater(logits, tf.constant(.5, shape=logits.shape)), tf.int32)
        return f1_score(labels,pred, average='micro')
        

def train(model, train_inputs, train_labels):

    for batch_start in tqdm(list(range(0, len(train_inputs), model.batch_size)), desc="training"):
        inputs = train_inputs[batch_start:min(batch_start + model.batch_size, len(train_inputs))]
        labels = train_labels[batch_start:min(batch_start + model.batch_size, len(train_labels))]

        with tf.GradientTape() as tape:
            logits = model.call(inputs)
            loss = model.loss(logits, labels)

        if batch_start % 2500 == 0:
            print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
    acc = []

    for batch_start in tqdm(list(range(0, len(test_inputs), model.batch_size)), desc="testing"):
        inputs = test_inputs[batch_start:min(batch_start + model.batch_size, len(test_inputs))]
        labels = test_labels[batch_start:min(batch_start + model.batch_size, len(test_labels))]
        logits = model.call(inputs)
        acc.append(model.accuracy(logits,labels))

    return sum(acc)/len(acc)



def main():
    window_size = 50
    train_inputs, train_labels, test_inputs, test_labels, vocab_len, num_genres, pad_id = get_data(window_size)
    # print(train_inputs.shape)
    # print(test_inputs.shape)
    # print(train_labels.shape)
    # print(test_labels.shape)
    

    model = Model(vocab_len, num_genres, window_size, pad_id)
    epochs = 5
    
    for e in range(epochs):
        print("Accuracy: ", test(model, test_inputs, test_labels))
        train(model, train_inputs, train_labels)


if __name__ == '__main__':
    main()
