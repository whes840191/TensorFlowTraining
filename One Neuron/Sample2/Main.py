import tensorflow as tf
import numpy as np

#Create a 1x2 matrix.
matrix1 = tf.constant([[3,3]])
#Create a 2x1 matrix.
matrix2 = tf.constant([[2],
                       [2]])

#Multiply matrix1 and matrix2 will create a object.
multuply = tf.matmul(matrix1, matrix2)

#We will calculate here.
with tf.Session() as sess:
    print(sess.run(multuply))