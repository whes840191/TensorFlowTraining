import tensorflow as tf
import numpy as np

def addNewLayer(inputs, in_size, out_size, activation_function = None):
    #uppercase mean a matrix variables.
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#from -1 to 1. one input and 300 training data.
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
#format is like as x_data
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
#excepted data.
y_data = np.square(x_data) - 0.5 + noise

#None mean any input matrix.
xs = tf.placeholder(tf.float32, [None,1])
ys = tf.placeholder(tf.float32, [None,1])

#input layer is 1 neuron, hidden layer 10 neuons, output layer 1 neuron
l1 = addNewLayer(xs, 1, 10 , activation_function=tf.nn.relu)
#Output layer no need the activation function.
prediction = addNewLayer(l1, 10, 1, activation_function=None)

#reduction_indices
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

#0.1 is learning rate, and we want to reduce the loss value.
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))