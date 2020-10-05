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
# Progressive GAN network

def pro_generator(x, res_building, res_training, latent_size):
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

    def grow(x, res_increase, res_decrease):
        y = block(x, res_increase)
        img = lambda: layers.upscale2d(to_rgb(y, res_increase-2), 2**res_decrease)
        if res_increase > 2:
            img = utils.cset(
                img,
                (res_training < res_increase),
                lambda: layers.upscale2d(utils.lerp(to_rgb(y, res_increase-2), layers.upscale2d(to_rgb(x, res_increase-3)), res_increase-res_training), 2**res_decrease))
        if res_decrease > 0:
            img = utils.cset(
                img,
                (res_training > res_increase),
                lambda: grow(y, res_increase + 1, res_decrease - 1))
        return img()

    outputs = grow(x, 2, res_building - 2)
    return outputs

def pro_discriminator(inputs, res_building, res_training, latent_size):
    def BA(x):
        x = layers.bias(x)
        x = layers.leaky_relu(x)
        return x
    def from_rgb(x, number):
        with tf.compat.v1.variable_scope('Input_{}'.format(number), reuse=tf.compat.v1.AUTO_REUSE):
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

    def grow(res_increase, res_decrease):
        x = lambda: from_rgb(layers.downscale2d(inputs, 2**res_decrease), res_increase-2)
        if res_decrease > 0:
            x = utils.cset(
                x,
                (res_training > res_increase),
                lambda: grow(res_increase+1, res_decrease-1))
        x = block(x(), res_increase); y = lambda: x
        if res_increase > 2:
            y = utils.cset(
                y,
                (res_training < res_increase),
                lambda: utils.lerp(x, from_rgb(layers.downscale2d(inputs, 2**(res_decrease+1)), res_increase-3), res_increase-res_training))
        return y()
        
    x = grow(2, res_building - 2)
    return x

def pro_gan(image_inputs, noise_inputs, latent_size, res_building, res_training, minibatch_size):
    # Define networks
    generator = network.Network('generator', pro_generator, noise_inputs, res_building=res_building, res_training=res_training, latent_size=latent_size)
    discriminator = network.Network('discriminator', pro_discriminator, image_inputs, res_building=res_building, res_training=res_training, latent_size=latent_size)
    # Retrieve network outputs
    fake_images = generator(noise_inputs)
    fake_outputs = discriminator(fake_images)
    real_outputs = discriminator(image_inputs)
    # Losses
    gen_loss, disc_loss = losses.RelativisticAverageBCE(real_outputs, fake_outputs)
    disc_loss += losses.GradientPenalty(discriminator,image_inputs,fake_images,minibatch_size)
    disc_loss += losses.EpsilonPenalty(real_outputs)
    return gen_loss, disc_loss, fake_images

#----------------------------------------------------------------------------
