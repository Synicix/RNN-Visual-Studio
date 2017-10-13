# LSTM

import tensorflow as tf
import numpy as np

# RNN Hyper Parameters
alphabet_size = 127
batch_size = 20
num_of_hidden_states = 10
num_of_steps = 5
learning_rate = 0.1
num_of_epocs = 10

def parseTextFile(file_path):
    words = []
    labels = []


    # Read
    with open(file_path) as fp:
        for line in fp:
            char_array = list(line)
            for char in char_array: # For every Char
                words.append(ord(char))
                labels.append(ord(char))



    labels.pop(1)

    return words, labels



def train_network(words, labels_key, num_epochs, num_steps = 5, state_size=4, verbose=True):
    session = tf.InteractiveSession()

    # Input
    with tf.name_scope("Inputs"):
        inputs = tf.placeholder(tf.int32, [None, num_of_steps, alphabet_size], name="inputs")
        labels = tf.placeholder(tf.int32, [None, alphabet_size], name="labels")

    rnn_inputs = tf.one_hot(inputs, 127, 1.0, 0.0, -1)
    rnn_labels = tf.one_hot(labels, 127, 1.0, 0.0, -1)

    # RNN
    init_state = tf.zeros([batch_size, num_of_hidden_states])

    with tf.name_scope("RNN_Cell"):
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_of_hidden_states)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, rnn_inputs, initial_state=init_state)

    # Summary
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/TensorBoard/RNN", session.graph)

    # Loss and Training Step
    with tf.variable_scope("Cell-Output"):
        W_y = weight_variable([alphabet_size, num_of_hidden_states], "W_y")
        b_y = bias_variable([alphabet_size], "b_y")

    logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W) + b,
                        [batch_size, num_steps, alphabet_size])
    predictions = tf.nn.softmax(logits)

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

    for i in range(2000):
        session.run(train_step, feed_dict={inputs: words, labels: labels_key})


def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)


def main():
    words, labels = parseTextFile("input.txt")
    train_network(words, labels, 10, 15)



if __name__ == "__main__":
    main()