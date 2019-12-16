import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from get_data import get_binary_data
import numpy as np
import sys
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import f1_score

from json import load
from itertools import chain, combinations


class Model(tf.keras.Model):
    def __init__(self, vocab_size, window_size):

        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = window_size
        self.embedding_size = 300
        self.hidden_size1 = 256
        self.hidden_size2 = 128
        self.batch_size = 128

        self.E = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.W1 = tf.keras.layers.Dense(self.hidden_size1, activation=tf.keras.layers.LeakyReLU())
        self.W2 = tf.keras.layers.Dense(self.hidden_size2, activation=tf.keras.layers.LeakyReLU())
        self.out = tf.keras.layers.Dense(2, activation='softmax')
    
        self.epsilon = 0.00002
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.epsilon)


    def call(self, inputs):
        embeddings = self.E(inputs)
        output_state = tf.reduce_sum(embeddings, 1)

        hidden1 = self.W1(output_state)
        hidden2 = self.W2(hidden1)
        logits = self.out(hidden2)
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
    for batch_start in range(0, len(train_inputs), model.batch_size):
        batch_end = min(batch_start + model.batch_size, len(train_inputs))
        inputs = train_inputs[batch_start:batch_end]
        labels = train_labels[batch_start:batch_end]
        
        with tf.GradientTape() as tape:
            logits = model.call(inputs)
            loss = model.loss(logits, tf.argmax(labels, 1))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    acc = []

    for batch_start in range(0, len(test_inputs), model.batch_size):
        inputs = test_inputs[batch_start:min(batch_start + model.batch_size, len(test_inputs))]
        labels = test_labels[batch_start:min(batch_start + model.batch_size, len(test_labels))]
        logits = model.call(inputs)
        acc.append(model.accuracy(logits,labels))

    return sum(acc) / len(acc)



def main():
    assert len(sys.argv) == 3
    task_id = int(sys.argv[1]) - 1
    max_size = int(sys.argv[2])

    def powerset_max_size(starting_set, k):
        return chain.from_iterable(combinations(starting_set, r) for r in range(1, k + 1))

    def get_pairs_max_size(stuff, k):
        subsets = list(powerset_max_size(stuff, k))
        return filter(lambda c: len(set(c[0]) & set(c[1])) == 0, combinations(subsets, 2))

    with open('processed/genres.json', 'r') as genres_file:
        genres = load(genres_file).keys()
    
    all_pairs_max_size = list(get_pairs_max_size(sorted(genres), max_size))


    equiv_classes = all_pairs_max_size[task_id]
    from subprocess import run, PIPE
    finished = run(['bash', '-c', 'cat all_recorded_accuracies | cut -f 1 -d ":" | sort -u'], stdout=PIPE, stderr=PIPE).stdout.decode(encoding='UTF-8')
    if finished.find(str(equiv_classes)) > -1:
        return

    equiv_class_0 = set(equiv_classes[0])
    equiv_class_1 = set(equiv_classes[1])

    window_size = 80
    train_inputs, train_labels, test_inputs, test_labels, vocab_len = get_binary_data(window_size, equiv_class_0, equiv_class_1)

    if len(train_inputs) + len(train_labels) < 10000:
        exit(0)

    total_breakdown = np.sum(train_labels, axis=0) + np.sum(test_labels, axis=0)
    if min(total_breakdown) / max(total_breakdown) < 0.33:
        exit(0)

    model = Model(vocab_len, window_size)
    epochs = 20

    best_accuracy = 0
    
    for e in range(epochs):
        #print("Accuracy: ", test(model, test_inputs, test_labels))
        best_accuracy = max(best_accuracy, test(model, test_inputs, test_labels))
        train(model, train_inputs, train_labels)

    best_accuracy = max(best_accuracy, test(model, test_inputs, test_labels))
    #print("Accuracy: ", test(model, test_inputs, test_labels))

    run(['bash', '-c', 'echo "{}:{}" >> ~/course/cs1470/Movie-Genre-Classifier/all_recorded_accuracies'.format(equiv_classes, best_accuracy)], stdout=PIPE, stderr=PIPE)


if __name__ == '__main__':
    main()
