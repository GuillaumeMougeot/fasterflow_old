import tensorflow as tf

from fasterflow import layers

#----------------------------------------------------------------------------
# Resnet inspired by fast.ai courses

def fastai_resnet(inputs, training, nbof_labels, regularizer_rate=0, dropout_rate=0.2):
    def BAN(x, use_bias=True, use_act=True, use_norm=True):
        if use_bias: x = layers.bias(x, regularizer_rate=regularizer_rate)
        if use_act: x = layers.leaky_relu(x, alpha=0.2)

        if use_norm:
            regularizer=tf.keras.regularizers.l2(regularizer_rate)
            x = tf.layers.batch_normalization(x, training=training, momentum=0.9, gamma_regularizer=regularizer, beta_regularizer=regularizer)
        # if use_norm: x = layers.batch_norm(x, training=training, decay=0.99) # Slower
        return x
    def conv_layer(name, x, fmaps, kernel=3, strides=1, padding='SAME'):
        with tf.compat.v1.variable_scope('Conv2D_{}'.format(name)):
            x = layers.conv2d(x, fmaps=fmaps, kernel=kernel, strides=strides, padding=padding, regularizer_rate=regularizer_rate)
            x = BAN(x, use_bias=False)
        return x
    def dense_layer(name, x, fmaps):
        with tf.compat.v1.variable_scope('Dense_{}'.format(name)):
            x = layers.dense(x, fmaps=fmaps, regularizer_rate=regularizer_rate)
            x = BAN(x, use_bias=False, use_act=False, use_norm=False)
        return x
    def res_layer(name, x, fmaps, kernel=3, strides=1, padding='SAME'):
        with tf.compat.v1.variable_scope('Res{}'.format(name)):
            x = x+conv_layer(name, x, fmaps, kernel=3, strides=1, padding='SAME')
            regularizer=tf.keras.regularizers.l2(regularizer_rate)
            x = tf.layers.batch_normalization(x, training=training, momentum=0.9, gamma_regularizer=regularizer, beta_regularizer=regularizer)
        return x
    
    x = conv_layer('inputs', inputs, 16, 3, 1)
    fmaps = [32,64,128]
    for i in range(len(fmaps)):
        x = conv_layer(name=i*3, x=x, strides=2, fmaps=fmaps[i])
        x = res_layer(name=i*3+1, x=x, strides=1, fmaps=fmaps[i])
    if dropout_rate > 0:
        x = tf.cond(training, lambda: tf.nn.dropout(x, rate=dropout_rate), lambda: x, name='use_dropout')
    logit = dense_layer(name='logit', x=x, fmaps=nbof_labels)
    return logit

#----------------------------------------------------------------------------
