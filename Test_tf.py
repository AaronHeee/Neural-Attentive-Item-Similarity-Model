
import tensorflow as tf
import numpy as np

p = tf.placeholder(tf.float32, shape=[None,None])
shape = tf.shape(p)[0]

stack = tf.stack([shape,1,1])
a = tf.Variable(tf.zeros([3,1]))
exp = tf.exp(a)
b = tf.Variable(tf.ones([3,1])) #tf.sequence_mask([1,2,1],4,dtype=tf.float32)

c = tf.Variable(tf.ones([3,3]))

g = tf.constant([10,1,1],tf.int32,[3,1])

height = tf.shape(g)[0]

add = b + c
mat = add * p

d = tf.pow(a + b, -1)
f = d*c


aa = tf.Variable(np.random.uniform(size=(1, 5)))
bb = tf.placeholder(shape=[None, None], dtype=tf.float32)
batch_size = tf.shape(bb)[0]
cc = tf.tile(aa, tf.stack([batch_size, 1]))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print sess.run(b)
    print sess.run(cc, feed_dict={bb: np.array([1,2,3])[:,None]})
    # print sess.run(shape)
    # print sess.run(stack)

    # print sess.run(f)
    # print sess.run(add)
    # print sess.run(shape, feed_dict={p: np.array([[1,2],[3,4]])})