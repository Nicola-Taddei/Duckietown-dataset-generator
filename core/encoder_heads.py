import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Conv2D, Concatenate, DepthwiseConv2D, Activation


def exit_flow(inputs: tf.Tensor, filter_count: int = 256):
    _x = keras.Sequential(name="exit_flow",
                          layers=[
                              Conv2D(filter_count,
                                     kernel_size=1,
                                     strides=1,
                                     padding="same",
                                     use_bias=False),
                              BatchNormalization(),
                              Activation("relu"),
                          ])(inputs)

    return _x


def dpc_head(inputs: tf.Tensor, filter_per_branch: int = 256):
    def dpc_conv_op(inputs: tf.Tensor, rate: list, name: str):
        _x = keras.Sequential(name=name,
                              layers=[
                                  DepthwiseConv2D(kernel_size=3,
                                                  strides=1,
                                                  dilation_rate=rate,
                                                  padding="same",
                                                  use_bias=False),
                                  BatchNormalization(),
                                  Activation("relu"),
                                  Conv2D(filter_per_branch,
                                         kernel_size=1,
                                         strides=1,
                                         padding="same",
                                         use_bias=False),
                                  BatchNormalization(),
                                  Activation("relu"),
                              ])(inputs)

        return _x

    # depth 1
    _x = dpc_conv_op(inputs, rate=[1, 6], name="dpc_head_branch_1_6")

    # depth 2
    _x1 = dpc_conv_op(_x, rate=[18, 15], name="dpc_head_branch_18_15")
    _x2 = dpc_conv_op(_x, rate=[6, 21], name="dpc_head_branch_6_21")
    _x3 = dpc_conv_op(_x, rate=[1, 1], name="dpc_head_branch_1_1")

    # depth 3
    _x4 = dpc_conv_op(_x1, rate=[6, 3], name="dpc_head_branch_6_3")

    _x = Concatenate()([_x, _x1, _x2, _x3, _x4])

    _x = exit_flow(_x, filter_count=filter_per_branch)

    return _x


def basic_head(inputs: tf.Tensor, filter_per_branch: int = 256):
    _, width, height, _ = inputs.shape

    left_path = keras.Sequential(name="basic_head_pooling",
                                 layers=[
                                     AveragePooling2D(pool_size=(width,
                                                                 height)),
                                     Conv2D(filter_per_branch,
                                            kernel_size=1,
                                            strides=1,
                                            padding="same",
                                            use_bias=False),
                                     BatchNormalization(),
                                     Activation("relu"),
                                 ])(inputs)
    left_path = tf.compat.v1.image.resize(left_path, [width, height],
                                          align_corners=True)

    right_path = keras.Sequential(name="basic_head_conv",
                                  layers=[
                                      Conv2D(filter_per_branch,
                                             kernel_size=1,
                                             strides=1,
                                             padding="same",
                                             use_bias=False),
                                      BatchNormalization(),
                                      Activation("relu"),
                                  ])(inputs)

    _x = Concatenate()([left_path, right_path])
    _x = exit_flow(_x, filter_count=filter_per_branch)

    return _x
