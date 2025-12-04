import tensorflow as tf

# Define a constant tensor
hello = tf.constant("Hello, TensorFlow!")

# Start a TensorFlow session (for TF 1.x) or just evaluate the tensor (TF 2.x)
# TensorFlow 2.x executes eagerly by default
print(hello.numpy().decode())  # decode() converts bytes to string
