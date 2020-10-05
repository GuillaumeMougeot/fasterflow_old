import tensorflow as tf

from fasterflow import layers

#----------------------------------------------------------------------------
# Resnet inspired by https://github.com/auroua/InsightFace_TF
# Itself an re-implementation of https://github.com/deepinsight/insightface
# This network reached 80% accuracy on Cifar10 without any data augmentation 
# and 96.73% on LFW with a cosface loss and trained on VGGFACE2 (3.1M of images) for 22M of images (about 7 epochs)

def insightface_resnet(
    inputs,
    training,
    nbof_labels,
    regularizer_rate=0,
    dropout_rate=0.,
    fmaps       = [64,128,256,256,512],
    nbof_unit   = [1,2,3,5,2],
    strides     = [2,2,2,2,2]):
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
            r = batch_norm(x)
            r = conv_layer(name+'_0', r, fmaps)
            r = conv_layer(name+'_1', r, fmaps, strides=strides)
            return r+s

    # For Cifar10
    # fmaps = [32,64,128,256]
    # nbof_unit = [1,2,3,5]
    # strides   = [1,2,2,2]
    # For vggface2 with 112x112
    # 22 millions of parameters (for face recognition)
    # fmaps = [64,128,256,256,512]
    # nbof_unit = [1,2,3,5,2] 
    # strides = [2,2,2,2,2]

    # Inputs
    x = conv_layer('inputs', inputs, fmaps[0], 3, 1)
    # Middle layers
    for i in range(len(fmaps)):
        x = block_res('{}_{}'.format(i,0), x, fmaps[i], strides=strides[i])
        for j in range(nbof_unit[i]-1):
            x = block_res('{}_{}'.format(i,j+1), x, fmaps[i], strides=1)
    # Output layers
    x = layers.global_avg_pool(x)
    if dropout_rate > 0:
        x = tf.cond(training, lambda: tf.nn.dropout(x, rate=dropout_rate), lambda: x, name='use_dropout')
    logit = dense_layer(name='logit', x=x, fmaps=nbof_labels)
    return logit

#----------------------------------------------------------------------------
