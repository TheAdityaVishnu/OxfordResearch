#Using the entire dataset of TUM

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

import scipy
from scipy import io
import datetime


mat3 = scipy.io.loadmat('Test704.mat');
Testing = mat3['Testing'];
Class_Testing = mat3['Class_Testing3'];

Testing = np.ascontiguousarray(Testing.T);
print(Testing.shape)
#from Draft1 import *
#from ForDelta import *
#from ForDelta import *
from ForBoth import *

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 14, 98])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])


Global_Data = np.arange(0,18746)

def variable_summaries(var):
    """ Calculate a variety of TensorBoard summary values for each tensor.
        This code is taken from the TensorFlow tutorial for TensorBoard,
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def next_batch(num, data, labels):
    '''
   Return a total of `num` random samples and labels.
   '''
    global Global_Data
    np.random.shuffle(Global_Data)
    idx = Global_Data[:num]
    #print(idx)
    #print(idx.shape)
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    for i in range(0, num):
        #print("Globl Data",Global_Data)
        Temp_Data = np.delete(Global_Data, 0)
        #print("Temp Data is ",Temp_Data)
        Global_Data = Temp_Data
        #print("Globl Dataafter delete",Global_Data)

       #¢print('Shape of TEMP', Temp_Data.shape)
        #print("VAlue of i is ", i )
        #print('Shape of Global Data',Global_Data.shape)
    #print("----End loop- ---")
    #print(Global_Data.shape)
    del Temp_Data
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Hidden layer 1
with tf.name_scope('convolution1'):
    with tf.name_scope('weights'):
        W_conv1 = weight_variable([4, 20, 1, 32])
    with tf.name_scope('biases'):
        b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 14, 98, 1])
    with tf.name_scope('activation_relu'):
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        tf.summary.histogram('activations', h_conv1)

# Hidden layer 2
with tf.name_scope('convolution2'):
    with tf.name_scope('weights'):
        W_conv2 = weight_variable([4, 20, 32, 64])
    with tf.name_scope('biases'):
        b_conv2 = bias_variable([64])
    with tf.name_scope('activation_rel'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        tf.summary.histogram('activations', h_conv2)

# Densely Connected Layer
with tf.name_scope('DenselyConnected'):
    with tf.name_scope('weights'):
        W_fc1 = weight_variable([25 * 4 * 64, 1024])
    with tf.name_scope('biases'):
        b_fc1 = bias_variable([1024])
    with tf.name_scope('activation_rel'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 25 * 4 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        tf.summary.histogram("activations",h_fc1)

with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('ReadoutLayer'):
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(0, 1830):
    Testing[i] = Testing[i] - Mean_C_2

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    for i in range(187):
        batch1, batch2 = next_batch(100, Training, Class_Training)
        if i % 5 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch1, y_: batch2, keep_prob: 1.0})
            tf.summary.scalar("Training Accuracy", train_accuracy)
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch1, y_: batch2, keep_prob: 0.5})
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: Testing, y_: Class_Testing, keep_prob: 1.0}))
    filename = "./summary_log/run" + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%s")
    writer = tf.summary.FileWriter(filename, sess.graph)
    summaryMerged = tf.summary.merge_all()

