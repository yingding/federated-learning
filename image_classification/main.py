import nest_asyncio
nest_asyncio.apply()

import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from matplotlib import pyplot as plt

np.random.seed(0)


# load a non-i.i.d Federated data
# https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
# print train data length
print(len(emnist_train.client_ids))

print(emnist_train.element_type_structure)

example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0]
)

example_element = next(iter(example_dataset))

print(type(example_element['label']))
print(example_element['label'].numpy())

plt.imshow(example_element['pixels'].numpy(), cmap='gray', aspect='equal')
plt.grid(False)
_ = plt.show()



