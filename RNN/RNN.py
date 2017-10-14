# LSTM

import tensorflow as tf
import numpy as np
import random

# RNN Hyper Parameters
alphabet_size = 96
batch_size = 100
num_of_hidden_states = 100
num_of_steps = 50
learning_rate = 0.1
iterations = 2000



def parseTextFile(file_path):
    words = []
    labels = []


    # Read
    with open(file_path) as fp:
        for line in fp:
            char_array = list(line)
            for char in char_array: # For every Char
                if char == "\n":
                    words.append(95)
                    labels.append(95)
                else:
                    words.append(ord(char) - 32)
                    labels.append(ord(char) - 32)
               
    

    labels.pop(1)
    labels.append(0)

    return words, labels


def getBatch(words, labels):
    temp_words = [[0 for x in range(num_of_steps)] for y in range(batch_size)]
    temp_lables = [[0 for x in range(num_of_steps)] for y in range(batch_size)]
    current_letter = 0
    for i in range(0, batch_size):
        current_letter = random.randint(0, words.__len__())
        for j in range(0, num_of_steps):
            temp_words[i][j] = words[current_letter]
            temp_lables[i][j] = labels[current_letter]
            current_letter = (current_letter + 1) % words.__len__()

    return temp_words, temp_lables


def build_graph(words, labels_key):
    session = tf.InteractiveSession()



    # Input
    with tf.name_scope("Inputs"):
        inputs = tf.placeholder(tf.int32, [None, num_of_steps], name="inputs")
        labels = tf.placeholder(tf.int32, [None, num_of_steps], name="labels")

    rnn_inputs = tf.one_hot(inputs, 127, 1.0, 0.0, -1)

    # LSTM

    with tf.name_scope("LSTM_Cell"):
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_of_hidden_states)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, rnn_inputs, dtype=tf.float32)
 

    # Loss and Training Step
    with tf.variable_scope("Cell-Output"):
        W_y = weight_variable([num_of_hidden_states, alphabet_size ], "W_y")
        b_y = bias_variable([alphabet_size], "b_y")
        tf.summary.histogram("W_y", W_y)
        tf.summary.histogram("b_y", b_y)
        
    with tf.name_scope("Loss"):
        logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs, [-1, num_of_hidden_states]), W_y) + b_y, [batch_size, num_of_steps, alphabet_size])
        predictions = tf.nn.softmax(logits)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        total_loss = tf.reduce_mean(losses)
        train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

        tf.summary.scalar("loss", total_loss)

        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/TensorBoard/RNN", session.graph)


        saver = tf.train.Saver()
        saver.restore(session, "/RNN/model.ckpt")

        for i in range(iterations):
            session.run(tf.global_variables_initializer())
            session.run(train_step, feed_dict={inputs: words, labels: labels_key})
            if i % 10 == 0:
                print("Step: " + str(i))
                s = session.run(summary, feed_dict={inputs: words, labels: labels_key})
                writer.add_summary(s, i)

                # Sample for 1000 characters

               # output = session.run([logits], feed_dict={inputs: words, labels: labels_key})

                #temp = np.argmax(output[0][0][0])
                #print(output[0][0][0])

        save_path = saver.save(session, "/RNN/model.ckpt")





def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)


def main():
    words, labels = parseTextFile("input.txt")
    batched_words, batched_labels = getBatch(words, labels)

    train_step = build_graph(batched_words, batched_labels)
 


if __name__ == "__main__":
    main()