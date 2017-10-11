import tensorflow as tf

#We set the name for this variable, and need to set "name =" for it.
state = tf.Variable(0, name = 'counter')
#print(state.name)

one = tf.constant(1)

new_Value = tf.add(state, one)
update = tf.assign(state, new_Value)

#Call global_variables_initializer to replace the initialize_all_variables(), and the warning will disappear.
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(3):
        sess.run(update)
        #Need to use sess.run to show value, if use state then this can't show the value.
        print(sess.run(state))