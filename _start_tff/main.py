import nest_asyncio
nest_asyncio.apply()

import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

def main():
    np.random.seed(0)

    input: str = 'Hello, World!'
    encoding = 'utf-8'

    result = tff.federated_computation(lambda: input)()
    # print(type(result))
    # print(result)

    # use utf-8 decode for byte()
    # https://stackoverflow.com/questions/606191/convert-bytes-to-a-string
    output: str = result.decode(encoding)
    print(f"result is {output}")


if __name__ == "__main__":
    main()