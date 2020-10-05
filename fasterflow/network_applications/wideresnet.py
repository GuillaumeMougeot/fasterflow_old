import tensorflow as tf

from fasterflow import layers

#----------------------------------------------------------------------------
# WideResNet: https://arxiv.org/pdf/1605.07146.pdf

def wideresnet(
    inputs,
    training,
    nbof_labels,
    regularizer_rate=0,
    fmaps     = [160,320,640],
    nbof_unit = [4,4,4],
    strides   = [1,2,2],
    dropouts  = [0.,0.,0.]):
    def batch_norm(x):
        with tf.compat.v1.variable_scope('BN'):
            # x = layers.batch_norm(x, training=training, regularizer_rate=regularizer_rate)
            x = tf.compat.v1.layers.batch_normalization(x, training=training, momentum=0.99, gamma_regularizer=tf.keras.regularizers.l2(regularizer_rate), beta_regularizer=tf.keras.regularizers.l2(regularizer_rate))
            # x = tf.nn.l2_normalize(x, axis=-1)
            # if len(x.shape)>2: x = layers.pixel_norm(x)
            return x
    def block_basic(name, x, fmaps, strides, dropout_rate=0.0):
        with tf.compat.v1.variable_scope('Block_{}'.format(name)):
            if x.shape[-1]==fmaps:
                r = layers.leaky_relu(batch_norm(x))
                s = x if strides==1 else tf.nn.max_pool2d(x, ksize=1, strides=2, padding='SAME')
            else:
                x = layers.leaky_relu(batch_norm(x))
                with tf.compat.v1.variable_scope('Shortcut'):
                    s = layers.conv2d(x, fmaps, kernel=1, strides=strides, regularizer_rate=regularizer_rate)
            with tf.compat.v1.variable_scope('Conv2D_0'):
                r = layers.leaky_relu(batch_norm(layers.conv2d(r if x.shape[-1]==fmaps else x, fmaps=fmaps, kernel=3, strides=strides, regularizer_rate=regularizer_rate)))
            if dropout_rate>0:
                r = tf.cond(training, lambda: tf.nn.dropout(r, rate=dropout_rate), lambda: r, name='use_dropout')
            with tf.compat.v1.variable_scope('Conv2D_1'):
                r = layers.conv2d(r, fmaps=fmaps, kernel=3, regularizer_rate=regularizer_rate)
            return r+s

    # Inputs
    with tf.compat.v1.variable_scope('Conv2D_1'):
        x = layers.conv2d(inputs, fmaps=fmaps[0], kernel=3, regularizer_rate=regularizer_rate)
    # Middle layers
    for i in range(len(fmaps)):
        x = block_basic('{}_{}'.format(i,0), x, fmaps[i], strides=strides[i], dropout_rate=dropouts[i])
        for j in range(nbof_unit[i]-1):
            x = block_basic('{}_{}'.format(i,j+1), x, fmaps[i], strides=1, dropout_rate=dropouts[i])
    # Output
    with tf.compat.v1.variable_scope('Output'):
        x = layers.leaky_relu(batch_norm(x))
        x = layers.global_avg_pool(x)
        logit = layers.dense(x, fmaps=nbof_labels, regularizer_rate=regularizer_rate)
    return logit

#----------------------------------------------------------------------------

def wideresnet_se(inputs, training, nbof_labels, regularizer_rate=0):
    def batch_norm(x):
        with tf.compat.v1.variable_scope('BN'):
            # x = layers.batch_norm(x, training=training, regularizer_rate=regularizer_rate)
            x = tf.compat.v1.layers.batch_normalization(x, training=training, momentum=0.99, gamma_regularizer=tf.keras.regularizers.l2(regularizer_rate), beta_regularizer=tf.keras.regularizers.l2(regularizer_rate))
            # if len(x.shape)>2: x = layers.pixel_norm(x)
            return x
    def block_se(name, x, fmaps, se_ratio):
        squeezed_fmaps = max(1,int(se_ratio*fmaps))
        with tf.compat.v1.variable_scope('SEBlock_{}'.format(name)):
            with tf.compat.v1.variable_scope('Squeeze'):
                s = tf.compat.v1.reduce_mean(x, axis=[1,2], keepdims=True)
                s = tf.nn.swish(layers.conv2d(s, squeezed_fmaps, kernel=1))
            with tf.compat.v1.variable_scope('Excite'):
                s = tf.nn.sigmoid(layers.conv2d(s, fmaps, kernel=1))
            return s*x
    act = layers.leaky_relu
    # act = tf.nn.selu
    def block_basic(name, x, fmaps, strides, dropout_rate=0.0):
        with tf.compat.v1.variable_scope('Block_{}'.format(name)):
            if x.shape[-1]==fmaps:
                r = act(batch_norm(x))
                s = x if strides==1 else tf.nn.max_pool2d(x, ksize=1, strides=2, padding='SAME')
            else:
                x = act(batch_norm(x))
                with tf.compat.v1.variable_scope('Shortcut'):
                    s = layers.conv2d(x, fmaps, kernel=1, strides=strides)
            # r = block_se(name, r, fmaps, 0.25)
            with tf.compat.v1.variable_scope('Conv2D_0'):
                r = act(batch_norm(layers.conv2d(r if x.shape[-1]==fmaps else x, fmaps=fmaps, kernel=3, strides=strides)))
            if dropout_rate>0:
                r = tf.cond(training, lambda: tf.nn.dropout(r, rate=dropout_rate), lambda: r, name='use_dropout')
            with tf.compat.v1.variable_scope('Conv2D_1'):
                r = layers.conv2d(r, fmaps=fmaps, kernel=3)
            r = block_se(name, r, fmaps, 0.25)
            return r+s

    # Inputs
    with tf.compat.v1.variable_scope('Conv2D_1'):
        x = layers.conv2d(inputs, fmaps=16, kernel=3)
    # Middle layers
    fmaps = [160,320,640]
    nbof_unit = [4,4,4]
    # nbof_unit = [1,1,1]
    strides   = [1,2,2]
    dropouts  = [0.,0.,0.]
    # dropouts  = [0.025,0.05,0.1]
    # dropouts  = [0.1,0.2,0.3]
    # strides   = [2,2,2]
    # nbof_unit = [2,3,4,6,3]
    for i in range(len(fmaps)):
        x = block_basic('{}_{}'.format(i,0), x, fmaps[i], strides=strides[i], dropout_rate=dropouts[i])
        for j in range(nbof_unit[i]-1):
            x = block_basic('{}_{}'.format(i,j+1), x, fmaps[i], strides=1, dropout_rate=dropouts[i])
    # Output
    with tf.compat.v1.variable_scope('Output'):
        x = act(batch_norm(x))
        x = layers.global_avg_pool(x)
        logit = layers.dense(x, fmaps=nbof_labels)
    return logit

#----------------------------------------------------------------------------
