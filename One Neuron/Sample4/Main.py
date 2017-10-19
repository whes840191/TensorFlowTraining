import tensorflow as tf

#We can add a additional matrix limitation like [2,2](need input a 2x2 matrix to this placeholder).
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    #We set the input value here.
    print(sess.run(output,feed_dict={input1:[7.], input2:[2.]}))