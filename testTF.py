import tensorflow as tf

a = tf.Variable([[[1,2,3]]], tf.int32)

c = tf.shape(a)[0]

b = tf.concat([a,tf.tile(a, tf.stack([1,c,1]))],2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(b)