# LSTM

import tensorflow as tf
import numpy as np
import random

# RNN Hyper Parameters
alphabet_size = 127
batch_size = 5
num_of_hidden_states = 10
num_of_steps = 1
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


def getBatch(words, lables):
    temp_words = [[0 for x in range(num_of_steps)] for y in range(batch_size)]
    temp_lables = [[0 for x in range(num_of_steps)] for y in range(batch_size)]

    current_letter = 0
    for i in range(0, batch_size):
        current_letter = random.randint(0, words.__len__())
        for j in range(0, num_of_steps):
            temp_words[i][j] = words[current_letter]
            temp_lables[i][j] = lables[current_letter]
            print(words[current_letter])
            current_letter = (current_letter + 1) % words.__len__()

    return temp_words, temp_lables


def train_network(words, labels_key, num_epochs, num_steps = 5, state_size=4, verbose=True):
    session = tf.InteractiveSession()

    # Input
    with tf.name_scope("Inputs"):
        inputs = tf.placeholder(tf.int32, [None, num_of_steps], name="inputs")
        labels = tf.placeholder(tf.int32, [None, num_of_steps], name="labels")

    rnn_inputs = tf.one_hot(inputs, 127, 1.0, 0.0, -1)

    # RNN
    init_state = tf.zeros([batch_size, num_of_hidden_states])

    with tf.name_scope("RNN_Cell"):
        rnn_cell = tf.contrib.rnn.BasicRNNCell(num_of_hidden_states)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, rnn_inputs, initial_state=init_state)



    # Loss and Training Step
    with tf.variable_scope("Cell-Output"):
        W_y = weight_variable([num_of_hidden_states, alphabet_size ], "W_y")
        b_y = bias_variable([alphabet_size], "b_y")
        
    with tf.name_scope("Loss"):
        logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs, [-1, num_of_hidden_states]), W_y) + b_y, [batch_size, num_of_steps, alphabet_size])
        predictions = tf.nn.softmax(logits)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        total_loss = tf.reduce_mean(losses)
        train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

        tf.summary.scalar("loss", total_loss)

            # Summary
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/TensorBoard/RNN", session.graph)

    for i in range(2000):
        session.run(tf.global_variables_initializer())
        session.run(train_step, feed_dict={inputs: words, labels: labels_key})
        if i % 100 == 0:
            print("Step: " + str(i))
            s = session.run(summary, feed_dict={inputs: words, labels: labels_key})
            writer.add_summary(s, i)


    num_epochs = 1
    verbose = True



def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)


def main():
    words, labels = parseTextFile("input.txt")
    batched_words, batched_labels = getBatch(words, labels)
    train_network(batched_words, batched_labels, 10, 15)



if __name__ == "__main__":
    main()