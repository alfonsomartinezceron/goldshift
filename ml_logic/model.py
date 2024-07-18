import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import numpy as np

class ConstantLayer(Layer):
    def __init__(self, value, **kwargs):
        super(ConstantLayer, self).__init__(**kwargs)
        self.value = value

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.fill([batch_size, 1], self.value)


def initialize_model(input_shape = (10,)) -> Model:
    value = 5.0  # The constant value you want the model to return
    constant_layer = ConstantLayer(value)
    # input_shape = (10,)  # Input shape can be anything; it won't affect the output
    inputs = Input(shape=input_shape)
    outputs = constant_layer(inputs)

    model = Model(inputs, outputs)
    return model

# Compile the model
def compile_model(model: Model) -> Model:
    model.compile(optimizer='adam', loss='mse')
    return model

# Print the model summary
def model_summary(model: Model):
    model.summary()

# Test the model
def model_test(model: Model):
    result_dict = {}
    test_input = np.random.random((7, 10))  # Batch of 7 samples, each with 10 features
    predictions = model.predict(test_input)
    print(predictions)  # Result is 7 times 5 in a matrix.
    return predictions


if __name__ == "__main__":
    model = initialize_model()
    model = compile_model(model)
    model_summary(model)
    model_test(model)
