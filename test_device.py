import tensorflow as tf

# Check TensorFlow version
print("TensorFlow Version:", tf.__version__)

# List physical devices
from tensorflow.python.client import device_lib
print("Available Devices:", device_lib.list_local_devices())

# Check if GPU is being used
if tf.test.is_gpu_available():
    print("GPU is available and being used.")
else:
    print("GPU is not available. Check your setup.")