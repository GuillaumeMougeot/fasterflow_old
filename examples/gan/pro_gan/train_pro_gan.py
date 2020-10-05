import tensorflow as tf
from functools import partial
import os
from shutil import copyfile
from datetime import datetime
import skimage as sk
import numpy as np
from time import time

from fasterflow import train
from fasterflow import dataset
from fasterflow import utils
from fasterflow import evaluate

import config_pro_gan as config
import network_pro_gan as network 

#----------------------------------------------------------------------------
# Training schedule

def training_schedule(cur_img):
    """
    Return the current resolution depending of the current img.
    """
    stage_length = config.stage_length
    trans_length = config.trans_length
    phase = cur_img // (stage_length+trans_length) + 2
    if phase >= np.log2(config.image_size):
        return np.log2(config.image_size)
    if cur_img % (stage_length+trans_length) < stage_length:
        return phase
    else:
        a = (cur_img % (stage_length+trans_length) - stage_length)/trans_length
        return phase + a

#----------------------------------------------------------------------------
# Train gan

def train_pro_gan():
    """
    Train the generative adverserial network.
    """
    # Graph definition: inputs, model, loss, optimizer, initializer
    print('Graph building...')
    # Inputs
    image_inputs        = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None]+config.image_shape, name='image_inputs')
    noise_inputs        = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None,config.latent_size], name='noise_inputs')
    res_training        = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[], name='res_training')
    minibatch_size      = tf.compat.v1.placeholder(tf.compat.v1.int64, shape=[], name='minibatch_size')
    # Network
    gen_loss, disc_loss, output_images = getattr(network, config.network)(
        image_inputs,
        noise_inputs,
        latent_size=config.latent_size,
        minibatch_size=minibatch_size,
        res_building=int(np.log2(config.image_size)),
        res_training=res_training)
    var_list            = tf.compat.v1.global_variables() # Store the list of variables
    # Optimizer
    lr                  = tf.compat.v1.placeholder(tf.compat.v1.float32, name='lr')
    optimizer_disc      = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, **config.optimzer_kwargs)
    optimizer_gen       = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, **config.optimzer_kwargs)
    disc_vars           = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    gen_vars            = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    training_op_disc    = optimizer_disc.minimize(disc_loss, var_list=disc_vars, name='training_op_disc')
    training_op_gen     = optimizer_gen.minimize(gen_loss, var_list=gen_vars, name='training_op_gen')
    # Initializer
    init                = tf.compat.v1.global_variables_initializer()
    print('Done: graph built.')

    # Init dataset
    inputs,_ = getattr(dataset, config.data_initilize)(**config.data_initilize_kwargs)

    # Define the minibatch selector
    select_minibatch = partial(getattr(dataset, config.data_selector), inputs)

    # Saver
    saver = train.Saver(var_list,config.logs_path, {'gen_loss':gen_loss, 'disc_loss':disc_loss})

    # Save minibatch test
    minibatch = select_minibatch(crt_img=0, res=config.image_size, minibatch_size=config.nbof_test_sample)/2 +0.5
    saver.save_images(minibatch, 'test')

    # Time measurements
    graph_time = time()

    # Training --> use the configuration file
    print('Training...')
    with tf.compat.v1.Session() as sess:
        # Initialize
        init.run()

        # Restore former parameters
        if config.restore:
            print('Restoring weight stored in {}'.format(config.restore))
            saver.restore(sess, config.restore)

        # Training parameters
        noise_test  = 2*np.random.random((config.nbof_test_sample, config.latent_size))-1

        # Training ...
        cur_img = config.start_img
        while cur_img < config.end_img:
            # Change input dataset
            cur_res = training_schedule(cur_img)
            cur_minibatch_size = config.minibatch_size[int(np.ceil(cur_res))]
            minibatch = select_minibatch(crt_img=cur_img, res=2**int(np.ceil(cur_res)), minibatch_size=cur_minibatch_size)
            noises = 2*np.random.random((cur_minibatch_size, config.latent_size))-1

            feed_dict = {}
            feed_dict['image_inputs:0'] = minibatch
            feed_dict['noise_inputs:0'] = noises
            feed_dict['lr:0'] = config.learning_rate
            feed_dict['res_training:0'] = cur_res
            feed_dict['minibatch_size:0'] = cur_minibatch_size

            sess.run([training_op_disc], feed_dict=feed_dict)
            sess.run([training_op_gen], feed_dict=feed_dict)

            # Display time information
            if cur_img % config.img_per_images == 0:
                graph_time = time() - graph_time
                minutes = int(graph_time // 60)
                secondes = graph_time - minutes * 60
                print('{} kimgs: {:4d} minutes {:2f} secondes, {:2f} resolution'.format(cur_img//1000, minutes, secondes, cur_res))
                graph_time = time()
            # Save logs
            if cur_img % config.img_per_summary == 0:
                saver.save_summary(sess, feed_dict, cur_img)
            # Save images
            if cur_img % config.img_per_images == 0:
                feed_dict = {noise_inputs:noise_test, res_training:cur_res}
                outputs = output_images.eval(feed_dict=feed_dict)
                outputs = outputs / 2 + 0.5 # Re scale output images
                outputs = np.clip(outputs, 0, 1)
                saver.save_images(outputs, cur_img//1000)
            # Save model
            if cur_img % config.img_per_save == 0:
                saver.save_model(sess, cur_img//1000)
            # Update current image
            cur_img += cur_minibatch_size
        # Final Saving
        saver.save_model(sess, 'final')
        saver.close_summary()
    print('Done: training')

#----------------------------------------------------------------------------
# Test

if __name__=='__main__':
    exec(config.training_function)

#----------------------------------------------------------------------------