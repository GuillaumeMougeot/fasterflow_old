import tensorflow as tf

from fasterflow import layers

#----------------------------------------------------------------------------
# Shake-shake: https://github.com/xgastaldi/shake-shake
# NOTE: this implementation is not accurate enough and has to be improved

def shake_shake(inputs, training, nbof_labels, regularizer_rate=0):
    def batch_norm(x):
        with tf.compat.v1.variable_scope('BN'):
            # x = layers.batch_norm(x, training=training, regularizer_rate=regularizer_rate)
            x = tf.compat.v1.layers.batch_normalization(x, training=training, momentum=0.99, gamma_regularizer=tf.keras.regularizers.l2(regularizer_rate), beta_regularizer=tf.keras.regularizers.l2(regularizer_rate))
            # if len(x.shape)>2: x = layers.pixel_norm(x)
            return x
    def BAN(x, use_bias=True, use_act=True, use_norm=True):
        if use_bias: x = layers.bias(x, regularizer_rate=regularizer_rate)
        if use_act: x = layers.leaky_relu(x, alpha=0.2)
        if use_norm: x = batch_norm(x)
        return x
    def conv_layer(name, x, fmaps, kernel=3, strides=1, padding='SAME'):
        with tf.compat.v1.variable_scope('Conv2D_{}'.format(name)):
            x = layers.conv2d(x, fmaps=fmaps, kernel=kernel, strides=strides, padding=padding, regularizer_rate=regularizer_rate)
            x = BAN(x)
        return x
    def dense_layer(name, x, fmaps):
        with tf.compat.v1.variable_scope('Dense_{}'.format(name)):
            x = layers.dense(x, fmaps=fmaps, regularizer_rate=regularizer_rate)
            x = BAN(x, use_bias=False, use_act=False, use_norm=True)
        return x
    def block_res(name, x, fmaps, strides):
        with tf.compat.v1.variable_scope('Block_{}'.format(name)):
            if x.shape[-1]==fmaps:
                s = x if strides==1 else tf.nn.max_pool2d(x, ksize=1, strides=2, padding='SAME')
            else:
                s = conv_layer('shortcut', x, fmaps, kernel=1, strides=strides)
            with tf.variable_scope('R'):
                r = batch_norm(x)
                r = conv_layer(name+'_0', r, fmaps)
                r = conv_layer(name+'_1', r, fmaps, strides=strides)
            with tf.variable_scope('T'):
                t = batch_norm(x)
                t = conv_layer(name+'_0', t, fmaps)
                t = conv_layer(name+'_1', t, fmaps, strides=strides)

            alpha = tf.compat.v1.get_variable(
            name='alpha',
            shape=[3,1,1,r.shape[-1]],
            dtype=r.dtype,
            initializer=tf.compat.v1.initializers.constant(1/3),
            trainable=True)
            return alpha[0]*r + alpha[1]*t + alpha[2]*s
            # alpha = tf.cond(training, lambda: tf.random.uniform([1])[0], lambda: tf.constant(0.5))
            # return alpha*r + (1-alpha)*t + s

    # Inputs
    x = conv_layer('inputs', inputs, 32, 3, 1)
    # Middle layers
    fmaps = [32,64,128,256]
    # fmaps = [64,128,256]
    nbof_unit = [1,1,1,1]
    # nbof_unit = [1,1,1]
    strides   = [1,2,2,2]
    # strides   = [2,2,2]
    # nbof_unit = [2,3,4,6,3]
    for i in range(len(fmaps)):
        x = block_res('{}_{}'.format(i,0), x, fmaps[i], strides=strides[i])
        for j in range(nbof_unit[i]-1):
            x = block_res('{}_{}'.format(i,j+1), x, fmaps[i], strides=1)
    x = layers.global_avg_pool(x)
    x = tf.cond(training, lambda: tf.nn.dropout(x, rate=0.2), lambda: x, name='use_dropout')
    logit = dense_layer(name='logit', x=x, fmaps=nbof_labels)
    return logit