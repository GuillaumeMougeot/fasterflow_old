import tensorflow as tf

from fasterflow import layers

#----------------------------------------------------------------------------
# EfficientNet: https://arxiv.org/abs/1905.11946

def efficientnet(inputs, training, nbof_labels, regularizer_rate=0):
    def batch_norm(x):
        with tf.compat.v1.variable_scope('BN'):
            x = tf.layers.batch_normalization(x, training=training, momentum=0.9, gamma_regularizer=tf.keras.regularizers.l2(regularizer_rate), beta_regularizer=tf.keras.regularizers.l2(regularizer_rate))
        return x
    def BAN(x, use_bias=True, use_act=True, use_norm=True):
        if use_bias: x = layers.bias(x, regularizer_rate=regularizer_rate)
        if use_norm: x = batch_norm(x)
        if use_act: x = tf.nn.relu6(x)
        return x
    def conv_layer(name, x, fmaps, kernel=3, strides=1, padding='SAME', use_bias=True, use_act=True, use_norm=True):
        with tf.compat.v1.variable_scope('Conv2D_{}'.format(name)):
            x = layers.conv2d(x, fmaps=fmaps, kernel=kernel, strides=strides, padding=padding, regularizer_rate=regularizer_rate)
            x = BAN(x)
        return x
    def depthwise_layer(name, x, kernel=3, strides=1, padding='SAME'):
        with tf.compat.v1.variable_scope('DepthwiseConv2D_{}'.format(name)):
            x = layers.depthwise_conv2d(x, kernel=kernel, strides=strides, padding=padding, regularizer_rate=regularizer_rate)
            x = BAN(x)
        return x
    def dense_layer(name, x, fmaps):
        with tf.compat.v1.variable_scope('Dense_{}'.format(name)):
            x = layers.dense(x, fmaps=fmaps, regularizer_rate=regularizer_rate)
            x = BAN(x, use_bias=False, use_act=False, use_norm=True)
        return x
    def block_res(name, x, fmaps, strides=1, use_conv=False):
        f1,f3 = fmaps*5,fmaps
        with tf.compat.v1.variable_scope('Block_{}'.format(name)):
            if use_conv: # conv block
                s = conv_layer('shortcut', x, f3, kernel=1, strides=strides, use_act=False)
            else: # identity block
                s = x; strides = 1
            r = conv_layer(name+'_0', x, fmaps=f1, kernel=1)
            r = depthwise_layer(name+'_1', r, strides=strides)
            r = conv_layer(name+'_2', r, fmaps=f3, kernel=1, strides=1, use_act=False)
            return tf.nn.relu6(r+s)
    # Inputs
    x = conv_layer('inputs', inputs, 16, 3, 1)
    # Middle layers
    fmaps       = [16,24,40,80,112]
    nbof_unit   = [1,2,3,4,1]
    strides     = [1,2,2,2,1]
    for i in range(len(fmaps)):
        x = block_res('{}_{}'.format(i,0), x, fmaps[i], strides=strides[i], use_conv=True)
        for j in range(nbof_unit[i]-1):
            x = block_res('{}_{}'.format(i,j+1), x, fmaps[i])
    x = conv_layer('outputs', x, 112*6, 1, 1)
    x = depthwise_layer('outputs', x, strides=1)
    x = layers.global_avg_pool(x)
    logit = dense_layer(name='logit', x=x, fmaps=nbof_labels)
    return logit

#----------------------------------------------------------------------------
# Efficient net from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py

import numpy as np

# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def round_filters(filters, width_coefficient, depth_divisor, min_depth):
    """Round number of filters based on depth multiplier."""
    multiplier = float(width_coefficient)
    divisor = int(depth_divisor)
    min_depth = min_depth
    if not multiplier:
        return filters
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def round_repeats(repeats, depth_coefficient):
    """Round number of filters based on depth multiplier."""
    multiplier = depth_coefficient
    if not multiplier:
        return repeats
    return int(np.ceil(multiplier * repeats))

# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def SEBlock(name, input_filters, se_ratio, expand_ratio, data_format='NHWC'):
    num_reduced_filters = max(1, int(input_filters * se_ratio))
    filters = input_filters * expand_ratio

    spatial_dims = [2, 3] if data_format == 'NCHW' else [1, 2]
    def block(inputs):
        with tf.variable_scope('SEBlock'):
            x = tf.compat.v1.reduce_mean(inputs, axis=spatial_dims, keepdims=True)
            with tf.variable_scope('Conv2D_0'):
                x = layers.conv2d(x, num_reduced_filters, kernel=1)
                x = layers.bias(x)
                x = tf.nn.swish(x)
            # Excite
            with tf.variable_scope('Conv2D_1'):
                x = layers.conv2d(x, filters, kernel=1)
                x = layers.bias(x)
                x = tf.nn.sigmoid(x)
            return x*inputs

    return block

# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def MBConvBlock(name, input_filters, output_filters,
                kernel_size, strides,
                expand_ratio, se_ratio,
                id_skip, training):

    has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
    filters = input_filters * expand_ratio

    def block(inputs):
        with tf.variable_scope('MBConvBlock_{}'.format(name)):
            if expand_ratio != 1:
                with tf.variable_scope('Expand'):
                    x = layers.conv2d(inputs, filters, kernel=1)
                    x = layers.batch_norm(x, training)
                    x = tf.nn.swish(x)
            else:
                x = inputs
            with tf.variable_scope('Depthwise'):
                x = layers.depthwise_conv2d(x, kernel=kernel_size, strides=strides)
                x = layers.batch_norm(x, training)
                x = tf.nn.swish(x)

            if has_se:
                x = SEBlock(name, input_filters, se_ratio, expand_ratio)(x)

            # output phase
            with tf.variable_scope('Output'):
                x = layers.conv2d(x, output_filters, kernel=1)
                x = layers.batch_norm(x, training)

            if id_skip:
                if strides == 1 and (input_filters == output_filters):
                    x = x+inputs
        return x

    return block

class BlockArgs(object):
    def __init__(self, input_filters=None,
                 output_filters=None,
                 kernel_size=None,
                 strides=None,
                 num_repeat=None,
                 se_ratio=None,
                 expand_ratio=None,
                 identity_skip=True):

        self.input_filters = input_filters
        self.output_filters = output_filters
        self.kernel_size=kernel_size
        self.strides = strides
        self.num_repeat = num_repeat
        self.se_ratio = se_ratio
        self.expand_ratio = expand_ratio
        self.identity_skip = identity_skip

def get_default_block_list():
    DEFAULT_BLOCK_LIST = [
        BlockArgs(32, 16, kernel_size=3, strides=1, num_repeat=1, se_ratio=0.25, expand_ratio=1),
        BlockArgs(16, 24, kernel_size=3, strides=2, num_repeat=2, se_ratio=0.25, expand_ratio=6),
        BlockArgs(24, 40, kernel_size=5, strides=2, num_repeat=2, se_ratio=0.25, expand_ratio=6),
        BlockArgs(40, 80, kernel_size=3, strides=2, num_repeat=3, se_ratio=0.25, expand_ratio=6),
        BlockArgs(80, 112, kernel_size=5, strides=1, num_repeat=3, se_ratio=0.25, expand_ratio=6),
        BlockArgs(112, 192, kernel_size=5, strides=2, num_repeat=4, se_ratio=0.25, expand_ratio=6),
        BlockArgs(192, 320, kernel_size=3, strides=1, num_repeat=1, se_ratio=0.25, expand_ratio=6),
    ]
    return DEFAULT_BLOCK_LIST

def EfficientNet(inputs, training, nbof_labels,
                 width_coefficient=1.0,
                 depth_coefficient=1.0,
                 #dropout_rate=0.,
                 depth_divisor=8,
                 min_depth=None):

    block_args_list = get_default_block_list()

    with tf.variable_scope('Input'):
        x = layers.conv2d(inputs, 32, kernel=1, strides=1)
        x = layers.batch_norm(x, training)
        x = tf.nn.swish(x)

    # Blocks part
    for i, block_args in enumerate(block_args_list):
        assert block_args.num_repeat > 0

        # Update block input and output filters based on depth multiplier.
        block_args.input_filters = round_filters(block_args.input_filters, width_coefficient, depth_divisor, min_depth)
        block_args.output_filters = round_filters(block_args.output_filters, width_coefficient, depth_divisor, min_depth)
        block_args.num_repeat = round_repeats(block_args.num_repeat, depth_coefficient)

        # The first block needs to take care of stride and filter size increase.
        x = MBConvBlock('{}_0'.format(i),block_args.input_filters, block_args.output_filters,
                        block_args.kernel_size, block_args.strides,
                        block_args.expand_ratio, block_args.se_ratio,
                        block_args.identity_skip, training)(x)

        if block_args.num_repeat > 1:
            block_args.input_filters = block_args.output_filters
            block_args.strides = 1

        for j in range(block_args.num_repeat - 1):
            x = MBConvBlock('{}_{}'.format(i,j+1),block_args.input_filters, block_args.output_filters,
                            block_args.kernel_size, block_args.strides,
                            block_args.expand_ratio, block_args.se_ratio,
                            block_args.identity_skip, training)(x)

    # Head part
    with tf.variable_scope('HeadConv'):
        x = layers.conv2d(x, fmaps=round_filters(1280, width_coefficient, depth_coefficient, min_depth), kernel=1)
        x = layers.batch_norm(x, training)
        x = tf.nn.swish(x)
    with tf.variable_scope('HeadDense'):
        x = layers.global_avg_pool(x)
        logit = layers.dense(x=x, fmaps=nbof_labels)
    return logit

#----------------------------------------------------------------------------
