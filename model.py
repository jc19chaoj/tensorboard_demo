import os
import sys
import urllib
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf

LOGDIR = '/tmp/mnist_tutorial/'
GITHUB_URL ='https://raw.githubusercontent.com/mamcgrath/TensorBoard-TF-Dev-Summit-Tutorial/master/'

mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + 'data', one_hot=True)

urlretrieve(GITHUB_URL + 'labels_1024.tsv', LOGDIR + 'labels_1024.tsv')
urlretrieve(GITHUB_URL + 'sprite_1024.png', LOGDIR + 'sprite_1024.png')

def conv_layer(name, inputs, output_size, filter_size=5, stride=1, padding='same'):
    # number of channels of input pictures
    input_shape = inputs.get_shape()
    print(input_shape)
    input_height, input_width, input_channel = input_shape[1:]

    with tf.name_scope(name):
        # create 'weights' and 'biases' initializer
        w_init = tf.constant_initializer(np.random.rand(filter_size, filter_size, input_channel, output_size).astype(np.float32))
        b_init = tf.constant_initializer(np.zeros((1, output_size)).astype(np.float32))

        conv = tf.layers.conv2d(inputs, output_size, filter_size,
                        strides=stride,
                        padding=padding,
                        activation=tf.nn.relu,
                        kernel_initializer=w_init,
                        bias_initializer=b_init,
                        name=name
                )
        conv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
        print(conv_vars)
        tf.summary.histogram('weights', conv_vars[0])
        tf.summary.histogram('biases', conv_vars[1])
        tf.summary.histogram('activation', conv)
        return conv

def pool_layer(name, inputs, filter_size=[2,2], stride=2):
    return tf.layers.max_pooling2d(inputs,
                          [filter_size[0], filter_size[-1]],
                          [stride, stride],
                          padding='same',
                          name=name)

def fc_layer(name, inputs, output_units, activation=tf.nn.relu):
    input_units = inputs.get_shape()[-1]

    with tf.name_scope(name):
        w_init = tf.constant_initializer(np.random.rand(input_units, output_units).astype(np.float32))
        b_init = tf.constant_initializer(np.zeros((1, output_units)).astype(np.float32))
        fc = tf.layers.dense(inputs, output_units, activation=activation,
                            name=name,
                            kernel_initializer=w_init,
                            bias_initializer=b_init)
        fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
        print(fc_vars)
        tf.summary.histogram('weights', fc_vars[0])
        tf.summary.histogram('biases', fc_vars[1])
        tf.summary.histogram('activation', fc)

        return fc

def simple_cnn(learning_rate):
    tf.reset_default_graph()
    sess = tf.Session()

    # Setup placeholders, and reshape the data
    X = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    X_reshape = tf.reshape(X, [-1, 28, 28, 1])
    print("Inputs:",X_reshape)
    tf.summary.image('input', X_reshape, 3)
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
    # conv1
    conv1 = conv_layer('conv1', X_reshape, 10)
    pool1 = pool_layer('pool1', conv1)
    # conv2
    conv2 = conv_layer('conv2', pool1, 64)
    pool2 = pool_layer('pool2', conv2)
    # fc1
    flatten_con2 = tf.reshape(pool2, [-1, 7*7*64])
    fc1 = fc_layer('fc1', flatten_con2, 1024)
    tf.summary.histogram('fc1', fc1)
    # fc2
    fc2 = fc_layer('fc2', fc1, 10, activation=None)

    with tf.name_scope('cross_entropy'):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=fc2, labels=y), name='xent')
        tf.summary.scalar('xent', xent)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(fc2, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    summary_merge = tf.summary.merge_all()

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR+"lr_%.0E"%learning_rate)
    writer.add_graph(sess.graph)


    for i in range(2001):
        batch = mnist.train.next_batch(100)
        # check training accuracy every 5 iterations
        if i % 5 == 0:
            [train_accuracy, s] = sess.run(
                [accuracy, summary_merge],
                feed_dict={X: batch[0], y: batch[1]})
            writer.add_summary(s, i)
            print("Step:", i, "accuracy =", accuracy)
        # save model checkpoint every 500 iterations
        if i % 500 == 0:
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)

        sess.run(optimizer, feed_dict={X: batch[0], y: batch[1]})

def main():
    for learning_rate in [1E-3, 1E-4]:

        print('Starting run for %s' % learning_rate)

        # run cnn model
        simple_cnn(learning_rate)

    print('Done training!')
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

if __name__ == '__main__':
    main()
