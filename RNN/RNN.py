# LSTM
# Daniel Sitonic

import tensorflow as tf
import numpy as np
import random

words = []
labels = []
inital = False


def parseTextFile(file_path):
    speical_characters_remap_index = 95;
    # Read
    with open(file_path) as fp:
        for line in fp:
            char_array = list(line)
            for char in char_array:  # For every Char
                if char == '\n':
                    words.append(speical_characters_remap_index)
                    labels.append(speical_characters_remap_index)
                elif char == '\t':
                    words.append(speical_characters_remap_index + 1)
                    labels.append(speical_characters_remap_index + 1)
                elif char == '“':
                    words.append(speical_characters_remap_index + 2)
                    labels.append(speical_characters_remap_index + 2)
                elif char == '”':
                    words.append(speical_characters_remap_index + 3)
                    labels.append(speical_characters_remap_index + 3)
                elif char == '’':
                    words.append(speical_characters_remap_index + 4)
                    labels.append(speical_characters_remap_index + 4)
                else:
                    if ord(char) - 32 < 0 or ord(char) - 32 > 100:
                        print(ord(char))
                    words.append(ord(char) - 32)
                    labels.append(ord(char) - 32)


    labels.pop(1)
    labels.append(0)
    return words, labels


def getBatch(num_of_steps, batch_size):
    temp_words = np.zeros(shape=(batch_size, num_of_steps))
    temp_lables = np.zeros(shape=(batch_size, num_of_steps))
    current_letter = 0
    for i in range(0, batch_size):
        current_letter = random.randint(0, words.__len__() - 1)
        for j in range(0, num_of_steps):
            temp_words[i][j] = words[current_letter]
            temp_lables[i][j] = labels[current_letter]
            current_letter = (current_letter + 1) % words.__len__()

    return temp_words, temp_lables


def build_graph_and_train(session, alphabet_size, batch_size, num_of_hidden_states, num_of_steps, num_of_layers, learning_rate, iterations, num_of_sample_char):
    with tf.device('/gpu:0'):
        # Input
        with tf.name_scope("Inputs"):
            inputs = tf.placeholder(tf.int32, [None, num_of_steps], name="inputs")
            inputs_sample = tf.placeholder(tf.int32, [1, num_of_steps])
            labels = tf.placeholder(tf.int32, [None, num_of_steps], name="labels")

        rnn_inputs = tf.one_hot(inputs, alphabet_size, 1.0, 0.0, -1)
        rnn_inputs_sample = tf.one_hot(inputs_sample, alphabet_size, 1.0, 0.0, -1)

        # LSTM
        with tf.name_scope("LSTM_Cell"):
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_of_hidden_states, state_is_tuple=True)
            rnn_outputs, final_state = tf.nn.dynamic_rnn(rnn_cell, rnn_inputs, dtype=tf.float32)
            rnn_outputs_sample, final_state_sample = tf.nn.dynamic_rnn(rnn_cell, rnn_inputs_sample, dtype=tf.float32)

        # Loss and Training Step
        with tf.name_scope("Output_Layer"):
            W_y = weight_variable([num_of_hidden_states, alphabet_size], "W_y")
            b_y = bias_variable([alphabet_size], "b_y")
            tf.summary.histogram("Weight", W_y)
            tf.summary.histogram("bias", b_y)


        with tf.name_scope("Loss"):
            logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs, [-1, num_of_hidden_states]), W_y) + b_y,
                                [batch_size, num_of_steps, alphabet_size])
            logits_sample = tf.reshape(tf.matmul(tf.reshape(rnn_outputs_sample, [-1, num_of_hidden_states]), W_y) + b_y,
                                        [1, num_of_steps, alphabet_size])

            predictions = tf.nn.softmax(logits)
            predictions_sample = tf.nn.softmax(logits_sample)
            print(predictions_sample)

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            total_loss = tf.reduce_mean(losses)
            train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

            tf.summary.scalar("loss", total_loss)

        saver = tf.train.Saver()
        saver.restore(session, "/RNN/model.ckpt")

        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/TensorBoard/RNN", session.graph)

        #session.run(tf.global_variables_initializer())

        # Training


        for i in range(0, iterations):
            words, labels_key = getBatch(num_of_steps, batch_size)
            session.run(train_step, feed_dict={inputs: words, labels: labels_key})
            print("Step: " + str(i))

            if i % 100 == 0:
                s = session.run(summary, feed_dict={inputs: words, labels: labels_key})
                writer.add_summary(s, i)
            if i % 1000 == 0:
                save_path = saver.save(session, "/RNN/model.ckpt")
                starting_char = random.randint(0, alphabet_size - 1)
                temp_words = np.zeros(shape=(1, num_of_steps), dtype=int)
                temp_words[0][0] = starting_char
                print("Sampling Text:\n")
                if starting_char == 95:
                    print("", end="\n")
                else:
                    char_temp = chr(starting_char + 32)
                    print(char_temp, end='', flush=True)
                

                # Sample text
                currentIndex = 0
                for j in range(0, num_of_sample_char):
                    next_char = session.run([predictions_sample], feed_dict={inputs_sample: temp_words})
                    if currentIndex == num_of_steps - 1: # Stop when the array is full
                        for k in range(0, num_of_steps - 1):
                            temp_words[0][k] = temp_words[0][k + 1]
                    else:
                        currentIndex += 1
                    temp_words[0][currentIndex] = np.argmax((next_char[0][0][currentIndex]))

                    if temp_words[0][currentIndex] == 95:
                        print('', end="\n")
                    elif temp_words[0][currentIndex] == 96:
                        print('\t', end="")
                    elif temp_words[0][currentIndex] == 97:
                        print('“', end="\n")
                    elif temp_words[0][currentIndex] == 98:
                        print('”', end="\n")
                    elif temp_words[0][currentIndex] == 99:
                        print('’', end="\n")
                    else:
                        print(chr(temp_words[0][currentIndex] + 32), end='', flush=True)
                        
                    

                print("\n")
                iterations = iterations * 2



def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def main():
    config = tf.ConfigProto(allow_soft_placement=True)
    session = tf.Session(config=config)

    # RNN Hyper Parameters
    alphabet_size = 100
    batch_size = 100
    num_of_hidden_states = 512
    num_of_steps = 100
    num_of_layers = 3
    learning_rate = 0.6
    iterations = 100

    num_of_sample_char = 5000

    # Sample from data
    parseTextFile("input2.txt")
    batched_words, batched_labels = getBatch(num_of_steps, batch_size)

    build_graph_and_train(session, alphabet_size, batch_size, num_of_hidden_states, num_of_steps, num_of_layers, learning_rate, 10000, num_of_sample_char)

    session.close()


if __name__ == "__main__":
    main()