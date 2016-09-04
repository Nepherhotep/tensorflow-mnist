# restore trained data
import cv2
import scipy
import tensorflow as tf
import numpy as np
import os


def print_float(value):
    if value:
        return '%.1f' % value
    else:
        return ' . '


def print_int(value):
    if value:
        return ' '
    else:
        return 'X'

np.set_printoptions(formatter={'float': print_float,
                               'int': print_int})


import sys
sys.path.append('mnist')
import model

x = tf.placeholder("float", [None, 784])
sess = tf.Session()

with tf.variable_scope("simple"):
    y1, variables = model.simple(x)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/simple.ckpt")


def simple(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convolutional.ckpt")


def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


def print_rating(arr):
    for i, value in enumerate(arr):
        print('{}: {:.2%}'.format(i, value))


def main(sample_name):
    sample_path = os.path.join('samples/binary/', sample_name)
    arr = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    print(arr)
    # print(arr)
    input = ((255 - arr) / 255.0).reshape(1, 784)
    output1 = simple(input)
    output2 = convolutional(input)
    print('simple:')
    print_rating(output1)
    print('\n')
    print('convoltional:')
    print_rating(output2)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        sample_name = sys.argv[1]
    else:
        sample_name = '0-1.png'
    main(sample_name)