import tensorflow as tf 
import skimage as sk 
import matplotlib.pyplot as plt
import numpy as np
import os

from fasterflow import layers
from fasterflow import utils
from fasterflow import losses

#----------------------------------------------------------------------------
# Network wrapper

from inspect import signature, Parameter

# The main function of this wrapper is to seperate the network definition from
# its calling. It preserves the weights using the 'reuse' mode from 
# tf.variable_scope()
class Network:
    def __init__(self, name, definition, *args, **kwargs):
        # Network name
        self.name = name
        print('#'*100)
        print('# Network: {:87s} #'.format(self.name))
        print('#'*100)
        # Network definition
        self.definition = definition
        # Network arguments
        self.net_kwargs = dict()
        # Retrieve the default network arguments
        self.net_sig = signature(self.definition)
        for key, value in self.net_sig.parameters.items():
            if value.default==Parameter.default: self.net_kwargs[key]=None 
            else: self.net_kwargs[key]=value.default
        # Replace the *args and **kwargs
        self.define_net_args(*args, **kwargs)
        # Initialize the network
        with tf.compat.v1.variable_scope(self.name):
            self.definition(**self.net_kwargs)
        # Store weights
        self.weights = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        # Display them
        self.display_weights()

    def __call__(self, *args, **kwargs):
        # Replace the *args and **kwargs
        self.define_net_args(*args, **kwargs)
        # Call the network
        with tf.compat.v1.variable_scope(self.name, reuse=True):
            return self.definition(**self.net_kwargs)

    def define_net_args(self, *args, **kwargs):
        assert len(args)+len(kwargs)<=len(self.net_kwargs), '[Error] Too many arguments.'
        # Replace the *args
        net_kwargs_keys = list(self.net_kwargs.keys())
        for i in range(len(args)): self.net_kwargs[net_kwargs_keys[i]] = args[i]
        # Replace the **kwargs
        for key, value in kwargs.items():
            assert key in self.net_kwargs, '[Error] {} not in network keyword arguments.'.format(key)
            self.net_kwargs[key] = value
    
    def display_weights(self):
        print('{:80s}{:20s}'.format('Weight name','Shape'))
        print('='*100)
        nbof_param = 0
        for weight in self.weights:
            weight_name = weight.name[:weight.name.rfind(':')]
            if 'weight' in weight_name:
                print('{:80s}{:20s}'.format(weight_name, str(weight.shape.as_list())))
            nbof_param += np.prod(weight.shape.as_list())
        print('='*100)
        print('Number of parameters: {}'.format(nbof_param))
        print('='*100)
    
    def max_norm(self, r=1.0, axes=-1):
        """
        Apply max normalization (weight clipping) to all weights in the network.
        NOTE: RUN tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS) DURING TRAINING PHASE.
        """
        for w in self.weights:
            # TODO: for convolution, see if weight clipping is better for other axis
            clipped_w = tf.compat.v1.clip_by_norm(w, clip_norm=r, axes=axes)
            clip_w = tf.compat.v1.assign(w, clipped_w)
            tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, clip_w)

#----------------------------------------------------------------------------
# Tests

def dummy(inputs, regularizer_rate=0):
    def BAN(x):
        x = layers.bias(x, regularizer_rate=regularizer_rate)
        x = tf.nn.selu(x)
        return x
    def conv_layer(name, x, fmaps, kernel=3, strides=1, padding='SAME'):
        with tf.compat.v1.variable_scope('Conv2D_{}'.format(name)):
            x = layers.conv2d(x, fmaps=fmaps, kernel=kernel, strides=strides, padding=padding, regularizer_rate=regularizer_rate)
            x = BAN(x)
        return x
    def dense_layer(x, fmaps, name=0, use_bias=True):
        with tf.compat.v1.variable_scope('Dense_{}'.format(name)):
            x = layers.dense(x, fmaps=fmaps, regularizer_rate=regularizer_rate)
            if use_bias: x = layers.bias(x, regularizer_rate=regularizer_rate)
        return x
    
    x = conv_layer('inputs', x=inputs, fmaps=64, kernel=1)
    fmaps = [64,128,256]
    for i in range(len(fmaps)):
        x = conv_layer(name=i*2, x=x, fmaps=fmaps[i])
        x = conv_layer(name=i*2+1, x=x, fmaps=fmaps[i])
        if i<(len(fmaps)-1): x = layers.downscale2d(x)
    x = dense_layer(x, fmaps=1, name='0', use_bias=False)
    x = layers.alpha_dropout(x, rate=0.2)
    x = tf.compat.v1.nn.l2_normalize(x, axis=1)
    return x

def test_netwrapper():
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[None,32,32,3])
    res_train = tf.compat.v1.placeholder(tf.float32, shape=[])
    net = Network('dummy', dummy, inputs)
    test = net(inputs*2)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        feed_dict = {inputs:np.random.random((1,32,32,3)),res_train:2}
        sess.run(test,feed_dict)
        
if __name__=='__main__':
    test_netwrapper()

#----------------------------------------------------------------------------
