import tensorflow as tf

X = tf.reshape(tf.range(50, dtype=tf.float32), (10, 5))

print(X)

sample_input = tf.ones((3, 5))
fc_layer = tf.keras.layers.Dense(units=3, input_shape=(5,))
fc_layer(sample_input)  # Forward pass to initialize weights and biases

print('X dim: ', X.shape)
print('W dim: ', fc_layer.weights[0].shape)
print('b dim: ', fc_layer.bias.shape)

A = fc_layer(X)

print('A:', A)
print('A dim: ', A.shape)




