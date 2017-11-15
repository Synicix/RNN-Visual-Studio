# ML Homework 2
# Daniel Sitonic

import tensorflow as tf

def main():

  # Import MNIST data set
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

  # Create Session
  sess = tf.InteractiveSession()

  # TensorBoard
  writer = tf.summary.FileWriter("/TensorBoard");
  writer.add_graph(sess.graph);


  # Variables
  with tf.name_scope("Basic-SoftMax-Layer"):
        x = tf.placeholder(tf.float32, [None, 784]) # Input vector [NONE,784] where none is the number of images
        W = tf.Variable(tf.zeros([784, 10]), name="W") # Weight Vector [784, 10]
        b = tf.Variable(tf.zeros([10]), name="b") # Biases [10]
        tf.summary.histogram("W", W)
        tf.summary.histogram("b", b)

  y_ = tf.placeholder(tf.float32, [None, 10]) # Cross-Entropy Variable

  y = tf.nn.softmax(tf.matmul(x, W) + b) # Model where x is [NONE, 784] * [784, 10] + [NONE, 10] = [NONE, 10]
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

  tf.summary.scalar("cross_entropy", cross_entropy)

  # Training
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  # Launching Model

  tf.global_variables_initializer().run()

  # Run Training step for 1000
  # We are using stocastic gradient decent, because it can get really expensive we used all examples
  # Instead we take a 100 random data points, and send it to the training algorithm
  summary = tf.summary.merge_all()
  writer = tf.summary.FileWriter("/TensorBoard/Softmax", sess.graph)

  for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 10 == 0:
        s = sess.run(summary, feed_dict={x: batch_xs, y_: batch_ys})
        writer.add_summary(s, i)

  # Computing the Error
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print("Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == "__main__":
    main()