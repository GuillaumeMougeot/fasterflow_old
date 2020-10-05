import tensorflow as tf

from fasterflow import layers

#----------------------------------------------------------------------------
# Standard Deep Convolutional Neural Networks

def deep_cnn_v1(x, training, nbof_labels, regularizer_rate=0, fmaps=[32,64,128]):
    def conv_layer(name, x, fmaps):
        with tf.compat.v1.variable_scope('Conv_{}'.format(name)):
            x = tf.compat.v1.layers.batch_normalization(
                tf.nn.elu(layers.conv2d(x,fmaps=fmaps,kernel=3,strides=1,regularizer_rate=regularizer_rate)),
                training=training,
                momentum=0.99,
                gamma_regularizer=tf.keras.regularizers.l2(regularizer_rate),
                beta_regularizer=tf.keras.regularizers.l2(regularizer_rate))
            return x
    x = conv_layer('Input',x,fmaps[0])
    x = tf.nn.max_pool2d(x, 2, 2, 'SAME')
    for i in range(len(fmaps)):
        x = conv_layer('{}_{}'.format(i,0), x, fmaps[i])
        x = conv_layer('{}_{}'.format(i,1), x, fmaps[i])
        x = tf.nn.max_pool2d(x, 2, 2, 'SAME')
        x = tf.cond(training, lambda: tf.nn.dropout(x, rate=0.2+i*0.1), lambda: x)
    x = layers.global_avg_pool(x)
    with tf.compat.v1.variable_scope('Output'):
        logit = layers.dense(x, fmaps=nbof_labels)
    return logit

def deep_cnn_v2(inputs, training, nbof_labels, regularizer_rate=0, dropout_rate=0.):
    def BAN(x, use_bias=True, use_act=True, use_norm=True):
        if use_bias: x = layers.bias(x, regularizer_rate=regularizer_rate)
        if use_act: x = layers.leaky_relu(x, alpha=0.2)
        # if use_norm:
        #     regularizer=tf.keras.regularizers.l2(regularizer_rate)
        #     x = tf.layers.batch_normalization(x, training=training, momentum=0.9, gamma_regularizer=regularizer, beta_regularizer=regularizer)
        if use_norm: x = layers.batch_norm(x, training=training, decay=0.99, regularizer_rate=regularizer_rate)
        return x
    def conv_layer(name, x, fmaps, kernel=3, strides=1, padding='SAME'):
        with tf.compat.v1.variable_scope('Conv2D_{}'.format(name)):
            x = layers.conv2d(x, fmaps=fmaps, kernel=kernel, strides=strides, padding=padding, regularizer_rate=regularizer_rate)
            x = BAN(x, use_bias=False)
        return x
    def dense_layer(x, fmaps, name=0):
        with tf.compat.v1.variable_scope('Dense_{}'.format(name)):
            x = layers.dense(x, fmaps=fmaps, regularizer_rate=regularizer_rate)
            x = BAN(x, use_bias=False, use_act=False, use_norm=False)
        return x
    
    with tf.compat.v1.variable_scope('Conv2D_{}'.format('inputs')):
        x = layers.conv2d(inputs, fmaps=10, kernel=5, strides=1, padding='SAME', regularizer_rate=regularizer_rate)
    fmaps = [20,40,80,160]
    for i in range(len(fmaps)):
        x = conv_layer(name=i*2, x=x, strides=2, fmaps=fmaps[i])
    # x = tf.nn.avg_pool2d(x, ksize=2, strides=1, padding='VALID')
    if dropout_rate > 0:
        x = tf.cond(training, lambda: tf.nn.dropout(x, rate=dropout_rate), lambda: x, name='use_dropout')
    logit = dense_layer(x, fmaps=nbof_labels, name='logit')
    return logit

#----------------------------------------------------------------------------
