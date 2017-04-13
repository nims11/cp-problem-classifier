# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name

import argparse
import sys
import vectorizer
import pickle
import numpy as np

np.random.seed(102)

import tensorflow as tf

FLAGS = None

CONCERNED_TAGS = [
    'dp',
    'binary search'
]

LIMIT = 1000
BATCH_SIZE = 64
EPOCH = 20

def load_data(fname):
    with open(fname, 'rb') as data_file:
        return pickle.load(data_file)

def get_y_vec(tags):
    y_vec = [1 if tag in tags else 0 for tag in CONCERNED_TAGS]
    return y_vec

def batch_iter(data, batch_size=BATCH_SIZE, num_epochs=EPOCH, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_data_and_labels(fname, limit=LIMIT, split=0.8):
    dataset = load_data(fname)
    x_train, y_train = [], []
    x_test, y_test = [], []
    problems = list(set([data['problem'] for data in dataset]))
    np.random.shuffle(problems)
    np.random.shuffle(dataset)
    idx = int(0.8 * len(problems))
    train_problems, test_problems = set(problems[:idx]), set(problems[idx:])

    for data in dataset:
        y_vec = get_y_vec(data['tags'])
        if sum(y_vec) != 1:
            continue
        if data['problem'] in train_problems:
            x_train.append(vectorizer.simple_ascii_vectorizer(data['source'], limit))
            y_train.append(y_vec)
        else:
            x_test.append(vectorizer.simple_ascii_vectorizer(data['source'], limit))
            y_test.append(y_vec)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x, [-1, 1, LIMIT, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([1, 50, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([1, 50, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  final_dimen = (LIMIT // 4) * 64
  W_fc1 = weight_variable([final_dimen, 512])
  b_fc1 = bias_variable([512])

  h_pool2_flat = tf.reshape(h_pool2, [-1, final_dimen])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([512, len(CONCERNED_TAGS)])
  b_fc2 = bias_variable([len(CONCERNED_TAGS)])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  X_train, Y_train, X_test, Y_test = load_data_and_labels('./dataset.pickle')

  # Create the model
  x = tf.placeholder(tf.float32, [None, LIMIT])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, len(CONCERNED_TAGS)])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for idx, batch in enumerate(batch_iter(list(zip(X_train, Y_train)))):
      xx, yy = zip(*batch)
      if idx % 100 == 0:
        print('step %d' % idx)
        train_accuracy = accuracy.eval(feed_dict={
            x:xx, y_: yy, keep_prob: 1.0})
        print('step %d, training accuracy %g' % (idx, train_accuracy))
      train_step.run(feed_dict={x: xx, y_: yy, keep_prob: 0.5})

    # print('train accuracy %g' % accuracy.eval(feed_dict={
    #     x: mnist.train.images, y_: mnist.train.labels, keep_prob: 1.0}))
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: X_test, y_: Y_test, keep_prob: 1.0}))

if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])
