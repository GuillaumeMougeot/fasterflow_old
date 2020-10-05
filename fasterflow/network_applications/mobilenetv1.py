import tensorflow as tf

from fasterflow import layers

#----------------------------------------------------------------------------
# Mobilenetv1: https://arxiv.org/abs/1704.04861

def v5(inputs, labels, emb_size, training, nbof_labels, regularizer_rate=0):
    def batch_norm(x):
        with tf.compat.v1.variable_scope('BN'):
            x = tf.layers.batch_normalization(x, training=training, momentum=0.9, gamma_regularizer=tf.keras.regularizers.l2(regularizer_rate), beta_regularizer=tf.keras.regularizers.l2(regularizer_rate))
        return x
    def BAN(x, use_bias=True, use_act=True, use_norm=True):
        if use_bias: x = layers.bias(x, regularizer_rate=regularizer_rate)
        if use_norm: x = batch_norm(x)
        # if use_act: x = tf.nn.relu6(x)
        if use_act: x = layers.leaky_relu(x, alpha=0.2)
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
        with tf.compat.v1.variable_scope('Block_{}'.format(name)):
            if use_conv: # conv block
                s = conv_layer('shortcut', x, fmaps*2, kernel=1, strides=strides, use_act=False)
            else: # identity block
                # s = x; strides = 1
                s = tf.zeros_like(x); strides=1
            # r = conv_layer(name+'_0', x, fmaps=f1, kernel=1)
            r = depthwise_layer(name+'_0', x, strides=strides)
            r = conv_layer(name+'_1', r, fmaps=fmaps*2, kernel=1, strides=1, use_act=False)
            return tf.nn.relu6(r+s)
    # Inputs
    x = conv_layer('inputs', inputs, 32, 3, 1)
    # Middle layers
    # fmaps       = [64,128,256,512]
    # nbof_unit   = [3,4,6,3]
    # strides     = [1,2,2,2]
    fmaps       = [32,64,128,256]
    nbof_unit   = [1,2,4,2]
    strides     = [1,2,2,2]
    for i in range(len(fmaps)):
        x = block_res('{}_{}'.format(i,0), x, fmaps[i], strides=strides[i], use_conv=True)
        for j in range(nbof_unit[i]-1):
            x = block_res('{}_{}'.format(i,j+1), x, fmaps[i])
    # x = conv_layer('outputs', x, 112*6, 1, 1)
    # x = depthwise_layer('outputs', x, strides=1)
    print('Last layer shape: {}'.format(x.shape))
    # x = layers.global_avg_pool(x)
    logit = dense_layer(name='logit', x=x, fmaps=nbof_labels)
    return logit

#----------------------------------------------------------------------------
