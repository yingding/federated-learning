import tensorflow as tf

def main():
    if tf.test.gpu_device_name():
        print(tf.test.gpu_device_name())
    else:
        print(f"no gpu support")    


if __name__ == "__main__":
    main()