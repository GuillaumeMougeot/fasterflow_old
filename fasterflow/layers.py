import tensorflow as tf
import numpy as np

#----------------------------------------------------------------------------
# Scaling layers

def upscale2d(x, factor=2, data_format='NHWC'):
    assert data_format in ['NHWC', 'NCHW']
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.compat.v1.variable_scope('Upscale2D'):
        if data_format=='NHWC':
            s = x.shape
            x = tf.compat.v1.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
            x = tf.compat.v1.tile(x, [1, 1, factor, 1, factor, 1])
            x = tf.compat.v1.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        if data_format=='NCHW':
            s = x.shape
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

def downscale2d(x, factor=2, data_format='NHWC'):
    assert data_format in ['NHWC', 'NCHW']
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.compat.v1.variable_scope('Downscale2D'):
        return tf.compat.v1.nn.avg_pool(x, ksize=factor, strides=factor, padding='VALID', data_format=data_format)

def global_avg_pool(x,  data_format='NHWC'):
    assert data_format in ['NHWC', 'NCHW']
    assert len(x.shape)==4
    axis=[1,2] if data_format=='NHWC' else [2,3]
    with tf.compat.v1.variable_scope('GlobalAvgPool'):
        return tf.compat.v1.math.reduce_mean(x,axis=axis)

#----------------------------------------------------------------------------
# Normalization layers

# Batch normalization:
# IMPORTANT NOTE 1: If you want to use this operator, please add 
# "extra_training_op   = tf.get_collection(tf.GraphKeys.UPDATE_OPS)"
# in your training process.
# Global mean and average will then be updated.
# IMPORTANT NOTE 2: This operator is for now slower than tf.layers.batch_normalization
def batch_norm(x, training, epsilon=1e-3, decay=0.99, data_format='NHWC', regularizer_rate=0.):
    assert regularizer_rate >= 0
    regularizer=tf.keras.regularizers.l2(regularizer_rate) if regularizer_rate else None
    with tf.compat.v1.variable_scope('batch_normalization'):
        shape=x.shape[-1]

        global_mean = tf.compat.v1.get_variable(
        name='moving_mean',
        shape=shape,
        dtype=x.dtype,
        initializer=tf.compat.v1.initializers.zeros(),
        trainable=False)
        global_variance = tf.compat.v1.get_variable(
        name='moving_variance',
        shape=shape,
        dtype=x.dtype,
        initializer=tf.compat.v1.initializers.ones(),
        trainable=False)

        beta = tf.compat.v1.get_variable(
        name='beta',
        shape=shape,
        dtype=x.dtype,
        initializer=tf.compat.v1.initializers.zeros(),
        regularizer=regularizer,
        trainable=True)
        gamma = tf.compat.v1.get_variable(
        name='gamma',
        shape=shape,
        dtype=x.dtype,
        initializer=tf.compat.v1.initializers.ones(),
        regularizer=regularizer,
        trainable=True)

        axes = [0] if len(x.shape)==2 else [0,1,2]
        mean, variance = tf.nn.moments(x, axes=axes)
        
        update_mean = tf.compat.v1.assign(global_mean, decay*global_mean+(1-decay)*mean)
        update_variance = tf.compat.v1.assign(global_variance, decay*global_variance+(1-decay)*variance)
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, update_mean)
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, update_variance)

        def f_train():
            scale = gamma * tf.compat.v1.rsqrt(variance+epsilon)
            offset = beta - scale * mean
            return scale, offset
        
        def f_eval():
            scale = gamma * tf.compat.v1.rsqrt(global_variance+epsilon)
            offset = beta - scale * global_mean
            return scale, offset

        scale, offset = tf.cond(training, f_train, f_eval)
        return scale*x+offset

# Pixel normalization. by Tero Karras from NVIDIA
def pixel_norm(x, epsilon=1e-8, data_format='NHWC'):
    assert data_format in ['NHWC', 'NCHW']
    with tf.compat.v1.variable_scope('PixelNorm'):
        axis=3 if data_format=='NHWC' else 1
        return x * tf.compat.v1.rsqrt(tf.compat.v1.math.reduce_mean(tf.compat.v1.square(x), axis=axis, keepdims=True) + epsilon)

# Minibatch standard deviation. by Tero Karras from NVIDIA
def minibatch_stddev_layer(x, group_size=4, data_format='NHWC'):
    assert data_format in ['NHWC', 'NCHW']
    with tf.compat.v1.variable_scope('MinibatchStddev'):
        group_size = tf.compat.v1.minimum(group_size, tf.compat.v1.shape(x)[0]) # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                                             # [NCHW]  Input shape.
        y = tf.compat.v1.reshape(x, [group_size, -1, s[1], s[2], s[3]])         # [GMCHW] Split minibatch into M groups of size G.
        y = tf.compat.v1.cast(y, tf.compat.v1.float32)                          # [GMCHW] Cast to FP32.
        y -= tf.compat.v1.reduce_mean(y, axis=0, keepdims=True)                 # [GMCHW] Subtract mean over group.
        y = tf.compat.v1.reduce_mean(tf.compat.v1.square(y), axis=0)            # [MCHW]  Calc variance over group.
        y = tf.compat.v1.sqrt(y + 1e-8)                                         # [MCHW]  Calc stddev over group.
        y = tf.compat.v1.reduce_mean(y, axis=[1,2,3], keepdims=True)            # [M111]  Take average over fmaps and pixels.
        y = tf.compat.v1.cast(y, x.dtype)                                       # [M111]  Cast back to original data type.
        shape=[group_size, s[1], s[2], 1] if data_format=='NHWC' else [group_size, 1, s[2], s[3]]
        y = tf.compat.v1.tile(y, shape)                                         # [N1HW]  Replicate over group and pixels. # MODIFIED
        axis=3 if data_format=='NHWC' else 1
        return tf.compat.v1.concat([x, y], axis=axis)                           # [NCHW]  Append as new fmap.

#----------------------------------------------------------------------------
# Activation layers

def leaky_relu(x, alpha=0.2):
    with tf.compat.v1.name_scope('LeakyRelu'):
        alpha = tf.compat.v1.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.compat.v1.maximum(x * alpha, x)

#----------------------------------------------------------------------------
# Dropout layers

def alpha_dropout(x, rate):
    with tf.compat.v1.name_scope('AlphaDropout'):
        a = 1/np.sqrt((1-rate)*(1+3.09*rate))
        b = 1.758*a*rate
        x = tf.nn.dropout(x, rate=rate)*(1-rate)
        mask = tf.cast(tf.math.logical_not(tf.cast(x, tf.bool)), tf.float32) # Keep zeros
        x = a*(x-1.758*mask)+b
    return x

#----------------------------------------------------------------------------
# Trainable layers, this layers have weights to train

# Bias layer, Note: data_format will be removed in further version
def bias(x, regularizer_rate=0, data_format='NHWC'):
    assert regularizer_rate >= 0
    assert data_format in ['NHWC', 'NCHW']
    shape=[x.shape[-1]] if data_format=='NHWC' else [x.shape[1]]
    regularizer=tf.keras.regularizers.l2(regularizer_rate) if regularizer_rate else None
    b = tf.compat.v1.get_variable(
        name='bias',
        shape=shape,
        dtype=x.dtype,
        initializer=tf.compat.v1.initializers.zeros(),
        regularizer=regularizer,
        trainable=True)
    if len(x.shape) == 2:
        return x + b
    else:
        shape=[1, 1, 1, -1] if data_format=='NHWC' else [1, -1, 1, 1]
        return x + tf.compat.v1.reshape(b, shape)

# Dense layer
def dense(x, fmaps, regularizer_rate=0):
    assert regularizer_rate >= 0
    if len(x.shape) > 2:
        x = tf.compat.v1.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    regularizer=tf.keras.regularizers.l2(regularizer_rate) if regularizer_rate else None
    w = tf.compat.v1.get_variable(
        name='weight',
        shape=[x.shape[1].value, fmaps],
        dtype=x.dtype,
        initializer=tf.compat.v1.initializers.he_normal(),
        regularizer=regularizer,
        trainable=True)
    return tf.compat.v1.linalg.matmul(x, w)

# Convolution 2D layer
def conv2d(x, fmaps, kernel, strides=1, padding='SAME', regularizer_rate=0, data_format='NHWC', dilations=1):
    assert regularizer_rate >= 0
    assert data_format in ['NHWC', 'NCHW']
    assert kernel >= 1
    assert type(dilations)==int
    in_fmaps=x.shape[-1].value if data_format=='NHWC' else x.shape[1].value
    regularizer=tf.keras.regularizers.l2(regularizer_rate) if regularizer_rate else None
    w = tf.compat.v1.get_variable(
        name='weight',
        shape=[kernel, kernel, in_fmaps, fmaps],
        dtype=x.dtype,
        initializer=tf.compat.v1.initializers.he_normal(),
        regularizer=regularizer,
        trainable=True)
    if type(strides)==int:
        strides=[1,strides,strides,1] if data_format=='NHWC' else [1,1,strides,strides]
    return tf.compat.v1.nn.conv2d(x, w, strides=strides, padding=padding, data_format=data_format, dilations=dilations)

# Depthwise convolution 2D layer
def depthwise_conv2d(x, channel_multiplier=1, kernel=1, strides=1, padding='SAME', regularizer_rate=0, data_format='NHWC', dilations=1):
    assert regularizer_rate >= 0
    assert data_format in ['NHWC', 'NCHW']
    assert kernel >= 1
    assert type(dilations)==int
    in_fmaps=x.shape[-1].value if data_format=='NHWC' else x.shape[1].value
    regularizer=tf.keras.regularizers.l2(regularizer_rate) if regularizer_rate else None
    w = tf.compat.v1.get_variable(
        name='weight',
        shape=[kernel, kernel, in_fmaps, channel_multiplier],
        dtype=x.dtype,
        initializer=tf.compat.v1.initializers.he_normal(),
        regularizer=regularizer,
        trainable=True)
    if type(strides)==int:
        strides=[1,strides,strides,1] if data_format=='NHWC' else [1,1,strides,strides]
    return tf.compat.v1.nn.depthwise_conv2d(x, w, strides=strides, padding=padding, data_format=data_format, dilations=[dilations])

# Fused upscale2d + conv2d. by Tero Karras from NVIDIA
# Faster and uses less memory than performing the operations separately.
def upscale2d_conv2d(x, fmaps, kernel, strides=2, data_format='NHWC'):
    assert data_format in ['NHWC', 'NCHW']
    assert kernel >= 1 and kernel % 2 == 1
    in_fmaps=x.shape[-1].value if data_format=='NHWC' else x.shape[1].value
    w = tf.compat.v1.get_variable(
        name='weight',
        shape=[kernel, kernel, in_fmaps, fmaps],
        dtype=x.dtype,
        initializer=tf.compat.v1.initializers.he_normal(),
        trainable=True)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], x.shape[1] * 2, x.shape[2] * 2, fmaps] if data_format=='NHWC' else [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
    if type(strides)==int:
        strides=[1,strides,strides,1] if data_format=='NHWC' else [1,1,strides,strides]
    return tf.nn.conv2d_transpose(x, w, os, strides=strides, padding='SAME', data_format=data_format)

#----------------------------------------------------------------------------
# Tests the layers

def test_bias():
    inputs = tf.compat.v1.placeholder(tf.compat.v1.float32,shape=[10,10],name='inputs')
    x = bias(inputs)
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        print(sess.run(x,{inputs:np.random.rand(10,10)}))

if __name__=='__main__':
    test_bias()

#----------------------------------------------------------------------------
