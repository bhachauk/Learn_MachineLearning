import tensorflow as tf
T, F = 1., -1.
train_in = [
 [T, T],
 [T, F],
 [F, T],
 [F, F],
]
train_out = [
 [T],
 [T],
 [T],
 [F],
]

w1 = tf.Variable(tf.random_normal([2, 2]))
w1 = tf.Print(w1, [w1], message="This is Inited w1: ")

b1 = tf.Variable(tf.zeros([2]))
b1 = tf.Print(b1, [b1], message="This is Inited b1: ")

w2 = tf.Variable(tf.random_normal([2, 1]))
w2 = tf.Print(w2, [w2], message="This is Inited w2: ")

b2 = tf.Variable(tf.zeros([1]))
b2 = tf.Print(b2, [b2], message="This is Inited b2: ")


# Layer 1
out1 = tf.tanh(tf.add(tf.matmul(train_in, w1), b1))
out1 = tf.Print(out1, [out1], message="This is out1: ")

# Layer 2
# Activation Layer
out2 = tf.sigmoid(tf.add(tf.matmul(out1, w2), b2))
out2 = tf.Print(out2, [out2], message="This is out2: ")

error = tf.subtract(train_out, out2)
mse = tf.reduce_mean(tf.square(error))

# training object with Optimizer
train = tf.train.GradientDescentOptimizer(0.01).minimize(mse)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
err, target = 1, 0.01
epoch, max_epochs = 0, 5

while err > target and epoch < max_epochs:
   epoch += 1
   err, _ = sess.run([mse, train])
print ('epoch:', epoch, 'mse:', err)