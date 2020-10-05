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

import config_msg_gan as config
import network_msg_gan as network 

#----------------------------------------------------------------------------
# Train gan

def train_msg_gan():
    """
    Train the generative adverserial network.
    """
    # Graph definition: inputs, model, loss, optimizer, initializer
    print('Graph building...')
    # Inputs
    image_inputs        = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None]+config.image_shape, name='image_inputs')
    noise_inputs        = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None,config.latent_size], name='noise_inputs')
    # Network
    gen_loss, disc_loss, output_images = getattr(network, config.network)(
        image_inputs,
        noise_inputs,
        latent_size=config.latent_size,
        minibatch_size=config.minibatch_size,
        res_building=int(np.log2(config.image_size)))
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
            minibatch = select_minibatch(crt_img=cur_img, res=config.image_size, minibatch_size=config.minibatch_size)
            noises = 2*np.random.random((config.minibatch_size, config.latent_size))-1

            feed_dict = {}
            feed_dict['image_inputs:0'] = minibatch
            feed_dict['noise_inputs:0'] = noises
            feed_dict['lr:0'] = config.learning_rate

            sess.run([training_op_disc], feed_dict=feed_dict)
            sess.run([training_op_gen], feed_dict=feed_dict)

            # Display time information
            if cur_img % config.img_per_images == 0:
                graph_time = time() - graph_time
                minutes = int(graph_time // 60)
                secondes = graph_time - minutes * 60
                print('{} kimgs: {:4d} minutes {:2f} secondes'.format(cur_img//1000, minutes, secondes))
                graph_time = time()
            # Save logs
            if cur_img % config.img_per_summary == 0:
                saver.save_summary(sess, feed_dict, cur_img)
            # Save images
            if cur_img % config.img_per_images == 0:
                feed_dict = {noise_inputs:noise_test}
                for i in range(len(output_images)):
                    outputs = output_images[i].eval(feed_dict=feed_dict)
                    outputs = outputs / 2 + 0.5 # Re scale output images
                    outputs = np.clip(outputs, 0, 1)
                    saver.save_images(outputs, '{}-{}'.format(cur_img//1000,2**(i+2)))
            # Save model
            if cur_img % config.img_per_save == 0:
                saver.save_model(sess, cur_img//1000)
            # Update current image
            cur_img += config.minibatch_size
        # Final Saving
        saver.save_model(sess, 'final')
        saver.close_summary()
    print('Done: training')

#----------------------------------------------------------------------------
# Test

if __name__=='__main__':
    exec(config.training_function)

#----------------------------------------------------------------------------