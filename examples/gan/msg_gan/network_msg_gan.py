import tensorflow as tf 
import skimage as sk 
import matplotlib.pyplot as plt
import numpy as np
import os

from fasterflow import layers
from fasterflow import utils
from fasterflow import losses
from fasterflow import network

#----------------------------------------------------------------------------
# Multi-scale gradient GAN network

def msg_generator(x, res_building, latent_size):
    # res: resolution, 2: 4x4, 3: 8x8, 4: 16x16, ...
    def BAN(x):
        x = layers.bias(x)
        x = layers.leaky_relu(x)
        x = layers.pixel_norm(x)
        return x
    def conv_layer(x, number):
        with tf.compat.v1.variable_scope('Conv2D_{}'.format(number)):
            x = layers.conv2d(x, fmaps=latent_size, kernel=3)
            x = BAN(x)
        return x
    def dense_layer(x, number):
        with tf.compat.v1.variable_scope('Dense_0'):
            x = layers.dense(x, fmaps=latent_size*16)
            x = tf.compat.v1.reshape(x, [-1, 4, 4, latent_size])
            x = BAN(x)
        return x
    def to_rgb(x, number):
        with tf.compat.v1.variable_scope('Output_{}'.format(number), reuse=tf.compat.v1.AUTO_REUSE):
            x = layers.upscale2d(x, factor=2**(res_building - number))
            x = layers.conv2d(x, fmaps=3, kernel=1)
            x = layers.bias(x)
        return x
    def block(x, res):
        with tf.compat.v1.variable_scope('Block_{}'.format(res)):
            if res==2:
                x = dense_layer(x, number=0)
                x = conv_layer(x, number=0)
            else:
                x = layers.upscale2d(x)
                x = conv_layer(x, number='{}_0'.format(res))
                x = conv_layer(x, number='{}_1'.format(res))
            return x

    outputs = []
    for res in range(2,res_building+1):
        x = block(x, res)
        outputs += [to_rgb(x, res)]
    return outputs

def msg_discriminator(inputs, res_building, latent_size):
    def BA(x):
        x = layers.bias(x)
        x = layers.leaky_relu(x)
        return x
    def from_rgb(x, number):
        with tf.compat.v1.variable_scope('Input_{}'.format(number), reuse=tf.compat.v1.AUTO_REUSE):
            x = layers.downscale2d(x, factor=2**(res_building - number))
            x = layers.conv2d(x, fmaps=latent_size, kernel=1)
            x = BA(x)
        return x
    def conv_layer(x, number):
        with tf.compat.v1.variable_scope('Conv2D_{}'.format(number)):
            x = layers.conv2d(x, fmaps=latent_size, kernel=3)
            x = BA(x)
        return x
    def dense_layer(x, fmaps, number):
        with tf.compat.v1.variable_scope('Dense_{}'.format(number)):
            x = layers.dense(x, fmaps=fmaps)
            x = layers.bias(x)
            if fmaps > 1:
                x = layers.leaky_relu(x)
        return x
    def block(x, res):
        with tf.compat.v1.variable_scope('Block_{}'.format(res)):
            if res==2:
                x = layers.minibatch_stddev_layer(x)
                x = conv_layer(x, number=0)
                x = dense_layer(x, fmaps=latent_size, number=1)
                x = dense_layer(x, fmaps=1, number=0)
            else:
                x = conv_layer(x, number='{}_0'.format(res))
                x = conv_layer(x, number='{}_1'.format(res))
                x = layers.downscale2d(x)
        return x

    for res in range(res_building,1,-1):
        x = from_rgb(inputs[res-2],res) if res==res_building else tf.compat.v1.concat([from_rgb(inputs[res-2],res),x], axis=-1)
        x = block(x, res)
    return x

def msg_gan(image_inputs, noise_inputs, latent_size, res_building, minibatch_size):
    # Multi-scaled input images
    real_inputs = []
    for factor in [2**res for res in range(3,-1,-1)]:
        real_input = layers.downscale2d(image_inputs,factor=factor)
        real_input = layers.upscale2d(real_input,factor=factor)
        real_inputs += [real_input]
    # Define networks
    generator = network.Network('generator', msg_generator, noise_inputs, res_building=res_building, latent_size=latent_size)
    discriminator = network.Network('discriminator', msg_discriminator, real_inputs, res_building=res_building, latent_size=latent_size)
    # Retrieve network outputs
    fake_images = generator(noise_inputs)
    fake_outputs = discriminator(fake_images)
    real_outputs = discriminator(real_inputs)
    # Losses
    gen_loss, disc_loss = losses.RelativisticAverageBCE(real_outputs, fake_outputs)
    disc_loss += losses.GradientPenaltyMSG(discriminator,real_inputs,fake_images,minibatch_size)
    # disc_loss += losses.EpsilonPenalty(real_outputs)
    return gen_loss, disc_loss, fake_images

#----------------------------------------------------------------------------
