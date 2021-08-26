import tensorflow as tf
from tensorflow import keras
import tensorflow_federated as tff
from tensorflow.keras import layers, Sequential, losses, metrics, optimizers

# Load simulation data.
source, _ = tff.simulation.datasets.emnist.load_data()

def client_data(n):
    return source.create_tf_dataset_for_client(source.client_ids[n]).map(
        # tf.reshape with shape [-1] mean flatten into 1D
        # https://stackoverflow.com/questions/41778632/why-is-the-x-variable-tensor-reshaped-with-1-in-the-mnist-tutorial-for-tensorfl/41778973#41778973
        lambda e: (tf.reshape(e['pixels'], [-1]) , e['label'])
    ).repeat(10).batch(20)

# Pick a subset of client devices to participate in training.
train_data = [client_data(n) for n in range(3)]

# Wrap a Keras model for use with TFF.
def model_fn():
    model = Sequential([
        layers.Dense(units=10, activation=tf.nn.softmax, input_shape=(784,), kernel_initializer='zeros')     
    ])

    return tff.learning.from_keras_model(
        model,
        input_spec=train_data[0].element_spec,
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()]
    )


# Simulate a few rounds of training with the selected client devices.
trainer = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: optimizers.SGD(0.1)
)

# How to sync state object in federated learning
state = trainer.initialize()
for _ in range(10):
    state, metrics = trainer.next(state, train_data)
    print(metrics['train']['loss'])
