import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np


def critic_net(s_dims):
    # 输入为state，输出为v
    # 网络的创建需要优化
    inputs = layers.Input(shape=(s_dims,))
    out = layers.Dense(16, activation='relu')(inputs)
    out = layers.Dense(32, activation='relu')(out)
    out_v = layers.Dense(1)(out)

    model = tf.keras.Model(inputs, out_v)
    return model


NN = critic_net(2)
print(np.zeros((1,2)))
state = np.array([1, 2])
state = tf.convert_to_tensor(state)
print(state)
state = tf.expand_dims(state,axis=0)
print(state)

output = NN(state)
print(output)
