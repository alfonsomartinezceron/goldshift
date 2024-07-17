import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

class ConstantLayer(Layer):
    def __init__(self, value, **kwargs):
        super(ConstantLayer, self).__init__(**kwargs)
        self.value = value

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.fill([batch_size, 1], self.value)


value = 5.0  # The constant value you want the model to return
constant_layer = ConstantLayer(value)

input_shape = (10,)  # Input shape can be anything; it won't affect the output
inputs = Input(shape=input_shape)
outputs = constant_layer(inputs)

model = Model(inputs, outputs)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Print the model summary
model.summary()

# Test the model
import numpy as np

test_input = np.random.random((7, 10))  # Batch of 7 samples, each with 10 features
predictions = model.predict(test_input)
print(predictions)  # Result is 7 times 5 in a matrix.
