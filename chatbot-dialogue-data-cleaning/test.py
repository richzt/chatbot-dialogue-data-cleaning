# encoding:utf-8
import numpy as np
x = np.array([[0.1], [0.8], [0.5]])
xx = np.array([[0], [1], [1]])
# if x.any():
#     print("yes")
# else:
#     print("no")
#
# a = x[:,  np.newaxis]   # row to columns
# print(a)
#
# xx = np.where(x > 0.5, 1, 0)
# print(xx)

import tensorflow as tf
sess = tf.Session()
with sess.as_default():
    x = tf.convert_to_tensor(x, dtype=np.float32)
    xx = tf.convert_to_tensor(xx, dtype=np.int32)
    y = tf.constant(0.5)
    t = tf.fill(tf.shape(x), 1)
    f = tf.fill(tf.shape(x), 0)
    print(tf.greater(x, y).eval())
    a = tf.where(tf.greater(x, y), t, f)
    a_n = a.eval()
    print(a_n)

    correct_prediction = tf.equal(a, xx)
    print(correct_prediction.eval())
