import tensorflow as tf
from functools import partial
import os
from shutil import copyfile
from datetime import datetime
import skimage as sk
import numpy as np
from time import time
import matplotlib.pyplot as plt

from fasterflow import train
from fasterflow import utils
from fasterflow import evaluate
from fasterflow import network
import data as dataset
from network_pro_gan import pro_generator, pro_discriminator
import config_pro_gan as config
from midi_utils import keyed_pianoroll2midi

#----------------------------------------------------------------------------
# Train gan

# restore = 'logs/music-pro_gan-run_20191027171540/models/model-7200.ckpt'
# restore = 'logs/music-pro_gan-run_20191028175516/models/model-8640.ckpt'
# restore = 'logs/music-pro_gan-run_20191028175516/models/model-480.ckpt'
restore = 'logs/music-pro_gan-run_20191028175516/models/model-9600.ckpt'
# restore = 'logs/music-pro_gan-run_20191028175516/models/model-11360.ckpt'
# restore = 'logs/music-pro_gan-run_20191028175516/models/model-final.ckpt'
latent_size = 256

def infer_pro_gan():
    """
    Train the generative adverserial network.
    """
    # Graph definition: inputs, model, loss, optimizer, initializer
    print('Graph building...')
    # Inputs
    noise_inputs        = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None,latent_size], name='noise_inputs')
    res_training        = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[], name='res_training')
    # Network
    generator           = network.Network('generator', pro_generator, noise_inputs, res_building=8, res_training=res_training, latent_size=latent_size)
    gen_vars            = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    outputs             = generator(noise_inputs)
    # outputs             = tf.math.reduce_mean(outputs, axis=-1)
    # outputs             = tf.clip_by_value(outputs, -1., 1.)
    # outputs             = outputs/2 + 0.5
    # outputs            *= 128

    # discriminator = network.Network('discriminator', pro_discriminator, image_inputs, res_building=res_building, res_training=res_training, latent_size=latent_size)

    # Initializer
    init                = tf.compat.v1.global_variables_initializer()
    print('Done: graph built.')

    # Saver
    saver =  tf.compat.v1.train.Saver(var_list=gen_vars)

    # Training --> use the configuration file
    print('Training...')
    with tf.compat.v1.Session() as sess:
        # Initialize
        init.run()
        # Restore parameters
        saver.restore(sess, restore)
        # Run
        n = 5
        out_tot = np.empty((0,256,256,1))
        for s in range(4):
            feed_dict = {
                noise_inputs:2*np.random.random((n*n, latent_size))-1,
                res_training:8
            }
            out = sess.run(outputs, feed_dict)
            out_tot = np.append(out_tot, out, axis=0)
            # for i in range(len(out)):
            #     im = out[i]*255
            #     im = im.astype(np.uint8)
            #     sk.io.imsave('logs/images/{}.jpg'.format(s*len(out)+i), im)
            # print(out.shape)
            # # fig = plt.figure()
            # # for i in range(n):
            # #     for j in range(n):
            # #         plt.subplot(n,n,i*n+j+1)
            # #         plt.imshow(out[i*n+j])
            # # plt.show()
            # # print(out[0,:5,:32])
            # for i in range(len(out)):
            #     # keyed_pianoroll2midi(out[i], file_path='../../../data/music/test_{}.mid'.format(i))
            #     # im = out[i]*255/np.max(out[i])
            #     # im = im.astype(np.uint8)
            #     keyed_pianoroll2midi(out[i]*100, file_path='logs/musics/{}.mid'.format(s*len(out)+i))
        name = 'logs/'+restore[restore.rfind('/')+1:] + '.npy'
        print(name)
        print(out_tot.shape)
        np.save(name, np.array(out_tot))

def infer_pro_disc():
    """
    Train the generative adverserial network.
    """
    # Graph definition: inputs, model, loss, optimizer, initializer
    print('Graph building...')
    # Inputs
    image_inputs        = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None,256,256,1], name='image_inputs')
    res_training        = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[], name='res_training')
    # Network
    discriminator = network.Network('discriminator', pro_discriminator, image_inputs, res_building=8, res_training=res_training, latent_size=latent_size)
    disc_vars           = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    outputs             = discriminator(image_inputs)
    outputs             = tf.nn.sigmoid(outputs)

    # Initializer
    init                = tf.compat.v1.global_variables_initializer()
    print('Done: graph built.')

    inputs = getattr(dataset, config.data_initilize)(**config.data_initilize_kwargs)
    select_minibatch = partial(getattr(dataset, config.data_selector), inputs)

    # Saver
    saver =  tf.compat.v1.train.Saver(var_list=disc_vars)

    # Training --> use the configuration file
    print('Training...')
    with tf.compat.v1.Session() as sess:
        # Initialize
        init.run()
        # Restore parameters
        saver.restore(sess, restore)
        # Run
        n = 20
        name = 'logs/'+restore[restore.rfind('/')+1:] + '.npy'
        out_tot = np.load(name)
        win = []
        
        for s in range(5):
            minibatch = select_minibatch(crt_img=n*s, res=8, minibatch_size=n)
            # minibatch = out_tot[n*s:n*(s+1),:,:]
            feed_dict = {
                image_inputs:minibatch,
                res_training:8
            }
            out = sess.run(outputs, feed_dict)
            win += [out] 
        print(np.mean((np.array(win).flatten()>0.5).astype(np.float32)))
        

def infer_pro_win(res = 2):
    """
    Train the generative adverserial network.
    """
    # Graph definition: inputs, model, loss, optimizer, initializer
    print('Graph building...')
    # Inputs
    noise_inputs        = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None,latent_size], name='noise_inputs')
    res_training        = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[], name='res_training')
    image_inputs        = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None,256,256,1], name='image_inputs')
    # Network
    generator           = network.Network('generator', pro_generator, noise_inputs, res_building=8, res_training=res_training, latent_size=latent_size)
    discriminator       = network.Network('discriminator', pro_discriminator, image_inputs, res_building=8, res_training=res_training, latent_size=latent_size)
    var_list            = tf.compat.v1.global_variables()
    gen_outputs         = generator(noise_inputs)
    fake_outputs        = discriminator(gen_outputs)
    real_outputs        = discriminator(image_inputs)
    rf = real_outputs - tf.compat.v1.math.reduce_mean(fake_outputs)
    fr = fake_outputs - tf.compat.v1.math.reduce_mean(real_outputs)
    # outputs             = tf.math.reduce_mean(outputs, axis=-1)
    # outputs             = tf.clip_by_value(outputs, -1., 1.)
    # outputs             = outputs/2 + 0.5
    # outputs            *= 128

    # Initializer
    init                = tf.compat.v1.global_variables_initializer()
    print('Done: graph built.')


    inputs = getattr(dataset, config.data_initilize)(**config.data_initilize_kwargs)
    select_minibatch = partial(getattr(dataset, config.data_selector), inputs)

    # Saver
    saver =  tf.compat.v1.train.Saver(var_list=var_list)

    # Training --> use the configuration file
    print('Training...')
    with tf.compat.v1.Session() as sess:
        # Initialize
        init.run()
        # Restore parameters
        saver.restore(sess, restore)
        # Run
        n = 4
        win = []
        for s in range(4):
            minibatch = select_minibatch(crt_img=n*n*s, res=8, minibatch_size=n*n)
            feed_dict = {
                noise_inputs:2*np.random.random((n*n, latent_size))-1,
                res_training:res,
                image_inputs:minibatch,
            }
            rf_, fr_ = sess.run([rf, fr], feed_dict)

            print(rf_)
            print(fr_)
            win = np.append(win, (rf_<.0).astype(np.float32))
            win = np.append(win, (fr_>.0).astype(np.float32))
            # out_tot += [out]
    print(np.mean(win))


if __name__=='__main__':
    # infer_pro_gan()
    # infer_pro_disc()
    infer_pro_win()

