import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)#Create 100  random constant input data.
#This is the excepted values for input x_data, so here are 100 constants.
y_data = x_data*0.1 + 0.3

# create tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0)) #dimenction is a [1] matrix, and range is from -1 to 1.
biases = tf.Variable(tf.zeros([1])) #set [1] as 0.

#Try to  train the Weights as 0.1 and biases as 0.3
#This is a real output when training, and this is a combination through Weights and biases, so it's could be a Tensorflow object.
y = Weights*x_data + biases

#Calculating the difference between the real output value and the excepted output value.
loss = tf.reduce_mean(tf.square(y-y_data))
#Choice a training method, and 0.5 is learning rate.
optimizer = tf.train.GradientDescentOptimizer(0.5)
#Use this train object to make the loss more small.
train = optimizer.minimize(loss)

#Create init all variable that we create before object.
init = tf.initialize_all_variables()
#Create tensorflow  structure end

#We can start train here.
# #Start init variables that we created.
sess = tf.Session()
sess.run(init)

for step in range(201):
    #Strat to train data through the optimizer
    sess.run(train)
    if step % 20 ==0:
        print(step, sess.run(Weights), sess.run(biases))