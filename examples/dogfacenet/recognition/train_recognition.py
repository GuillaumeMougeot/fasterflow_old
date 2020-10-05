import tensorflow as tf
from functools import partial
import os
from shutil import copyfile
from datetime import datetime
import skimage as sk
import numpy as np
from time import time

from fasterflow import dataset
from fasterflow import utils
from fasterflow import evaluate
from fasterflow import train
from fasterflow import layers
from fasterflow import prepare_data
from fasterflow import losses

import config_recognition as config
import network_recognition as network

from online_training import *

from fasterflow.network_applications.deep_cnn import deep_cnn_v1
from fasterflow.network_applications.wideresnet import wideresnet

#----------------------------------------------------------------------------
# Training schedule

def training_schedule(cur_img, decrease_rate=config.decrease_rate, decay=1.5):
    cur_stage = int(np.log2(cur_img//config.stage_length + 1))
    stage_start = (2**cur_stage-1)*config.stage_length
    stage_length = (2**cur_stage) * config.stage_length
    cur_img -= stage_start
    phase = ((decrease_rate-1)*(cur_img % stage_length)/stage_length + 1)*decay**cur_stage
    return config.learning_rate/phase

#----------------------------------------------------------------------------
# Train classification and recognition

def load_images(filenames):
    images = []
    for i in range(len(filenames)):
        img = sk.io.imread(filenames[i])
        # images += [sk.transform.resize(img, config.image_shape)]
        images += [img]
    return utils.downscale2d_np(images)

def train_triplet_from_images():
    """
    Train the classification or the recognition network
    mode is in ['classification', 'recognition']
    """
    # Graph definition: inputs, model, loss, optimizer, initializer
    print('Graph building...')
    # Inputs
    images       = tf.compat.v1.placeholder(tf.float32,shape=[None,config.image_size,config.image_size,3],name='image_inputs')
    # label_inputs = tf.compat.v1.placeholder(tf.int64, shape=[None,], name='label_inputs')
    training     = tf.placeholder(tf.bool, shape=[], name='training')
    lr           = tf.compat.v1.placeholder(tf.float32, shape=[], name='learning_rate')
    # Augmentation
    image_inputs = images/127.5 - 1
    aug = lambda: dataset.augment_image(
        image_inputs,
        config.minibatch_size,
        use_horizontal_flip=True,
        rotation_rate   =0.3,
        translation_rate=0.2,
        cutout_size     =25,
        crop_pixels     =10)
    image_inputs = tf.cond(training, aug, lambda: image_inputs)
    # Network
    emb,loss,reg_loss = network.recognition(image_inputs, config.emb_size, training, config.regularizer_rate)
    # Optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, **config.optimzer_kwargs)
    var_list  = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=config.network)
    assert len(var_list)>0
    # Training operations
    training_op       = optimizer.minimize(loss+reg_loss, var_list=var_list, name='training_op')
    extra_training_op = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS) # For batch normalization
    # Initializer
    init = tf.compat.v1.global_variables_initializer()
    print('Done: graph built.')

    # Prepare the dataset
    training_filenames, training_labels = prepare_data.prepare_data(config.data_path)
    validation_filenames, validation_labels = prepare_data.prepare_dogfacenet(config.data_test_path)

    # Saver
    with tf.compat.v1.variable_scope('Saver'):
        training_accuracy = losses.triplet_accuracy(emb)
        # Saver placeholders
        loss_saver = tf.compat.v1.placeholder(tf.float32, shape=[], name='loss_saver')
        training_accuracy_saver = tf.compat.v1.placeholder(tf.float32, shape=[], name='training_accuracy_saver')
        validation_accuracy_saver = tf.compat.v1.placeholder(tf.float32, shape=[], name='validation_accuracy_saver')
        saver = train.Saver(
            var_list,
            config.logs_path,
            summary_dict={
                'loss':loss_saver,
                'training_accuracy':training_accuracy_saver,
                'validation_accuracy':validation_accuracy_saver,
                'learning_rate':lr},
            restore=None)
    
    # Time measurements
    init_time = time()
    # Training --> use the configuration file
    print('Training...')
    with tf.compat.v1.Session() as sess:
        # Initialize
        init.run()
        # Restore former parameters
        if config.restore:
            print('Restoring weight stored in {}'.format(config.restore))
            saver.restore(sess, config.restore)

        # Training ...
        cur_img = config.start_img
        while cur_img < config.end_img:
            filenames_,_ = define_triplets_batch(training_filenames,training_labels,nbof_triplet=config.minibatch_size)
            
            images_ = load_images(filenames_)
            
            feed_dict = {
                images:   images_,
                training: True,
                lr:       training_schedule(cur_img)}

            loss_, training_accuracy_, _, _ = sess.run([loss, training_accuracy, training_op, extra_training_op], feed_dict=feed_dict)

            # Validation
            if cur_img % config.img_per_val == 0:
                pred_validation = np.empty((0,config.emb_size))
                labels_test = np.empty((0))
                for i in range(0,len(validation_labels),config.minibatch_size):
                    images_ = load_images(validation_filenames[i:i+config.minibatch_size])
                    feed_dict = {
                        images:   images_,
                        training: False}
                    pred_validation = np.append(pred_validation, sess.run(emb, feed_dict), axis=0)
                    labels_test = np.append(labels_test,validation_labels[i:i+config.minibatch_size])
                _, _, acc, val, _, far=evaluate.evaluate(pred_validation, labels_test[::2])
                validation_accuracy_ = np.mean(acc)
                feed_dict_saver = {
                    loss_saver: loss_,
                    training_accuracy_saver: training_accuracy_,
                    validation_accuracy_saver: validation_accuracy_,
                    lr:training_schedule(cur_img)}
                saver.save_summary(sess, feed_dict_saver, cur_img)
                # Display information
                graph_time  = time() - init_time
                hours       = int(graph_time // 3600)
                minutes     = int(graph_time // 60) - hours*60
                secondes    = graph_time - minutes * 60 - hours * 3600
                print('{:4d} kimgs, {}h {:2d}m {:2.1f}s, lr:{:0.6f}, training: {:2.2f}%, validation: {:2.2f}%'.format(
                    cur_img//1000,
                    hours, minutes,
                    secondes,
                    training_schedule(cur_img),
                    training_accuracy_*100,
                    validation_accuracy_*100))
            # Save logs
            if cur_img % config.img_per_summary == 0:
                feed_dict_saver = {
                    loss_saver: loss_,
                    training_accuracy_saver: training_accuracy_,
                    validation_accuracy_saver: validation_accuracy_,
                    lr:training_schedule(cur_img)}
                saver.save_summary(sess, feed_dict_saver, cur_img)
            # Save model
            if cur_img % config.img_per_save == 0:
                saver.save_model(sess, cur_img//1000)
            # Update current image
            cur_img += config.minibatch_size
        # Final Saving
        saver.save_model(sess, 'final')
        saver.close_summary()
    print('Done: training')

def train_triplet_from_npy():
    """
    Train the classification or the recognition network
    mode is in ['classification', 'recognition']
    """
    # Prepare the dataset
    training_filenames, training_labels_bytes = np.load(config.data_path)
    training_labels = np.array([int(tlb.decode('ascii')) for tlb in training_labels_bytes]) # Convert labels type: from bytes to int
    validation_filenames, validation_labels_bytes = np.load(config.data_test_path)
    validation_labels = np.array([int(vlb.decode('ascii')) for vlb in validation_labels_bytes]) # Convert labels type: from bytes to int

    # Graph definition: inputs, model, loss, optimizer, initializer
    print('Graph building...')
    # Small graph for decoding jpegs
    with tf.compat.v1.variable_scope('DecodeJPEG'):
        image_jpeg  = tf.compat.v1.placeholder(tf.string,shape=None,name='images_raw')
        image_decoded = tf.io.decode_jpeg(image_jpeg)
    # Inputs
    images       = tf.compat.v1.placeholder(tf.float32,shape=[None,config.image_size,config.image_size,3],name='image_inputs')
    label_inputs = tf.compat.v1.placeholder(tf.int64, shape=[None,], name='label_inputs')
    training     = tf.placeholder(tf.bool, shape=[], name='training')
    lr           = tf.compat.v1.placeholder(tf.float32, shape=[], name='learning_rate')
    # Augmentation
    image_inputs = images/127.5 - 1
    aug = lambda: dataset.augment_image(
        image_inputs,
        config.minibatch_size,
        use_horizontal_flip=False,
        rotation_rate   =0.3,
        translation_rate=0.2,
        cutout_size     =25,
        crop_pixels     =10)
    image_inputs = tf.cond(training, aug, lambda: image_inputs)
    # Network
    # emb,loss,reg_loss = network.recognition(image_inputs, config.emb_size, training, config.regularizer_rate)
    emb,loss,reg_loss = network.recognition_micheal(image_inputs, config.emb_size, label_inputs, config.minibatch_size, training, config.regularizer_rate)
    # Optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, **config.optimzer_kwargs)
    var_list  = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=config.network)
    assert len(var_list)>0
    # Training operations
    training_op       = optimizer.minimize(loss+reg_loss, var_list=var_list, name='training_op')
    extra_training_op = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS) # For batch normalization
    # Initializer
    init = tf.compat.v1.global_variables_initializer()
    print('Done: graph built.')

    # Saver
    with tf.compat.v1.variable_scope('Saver'):
        # training_accuracy = losses.triplet_accuracy(emb)
        training_accuracy = losses.general_triplet_accuracy(emb, label_inputs, config.minibatch_size)
        # Saver placeholders
        loss_saver = tf.compat.v1.placeholder(tf.float32, shape=[], name='loss_saver')
        training_accuracy_saver = tf.compat.v1.placeholder(tf.float32, shape=[], name='training_accuracy_saver')
        validation_accuracy_saver = tf.compat.v1.placeholder(tf.float32, shape=[], name='validation_accuracy_saver')
        saver = train.Saver(
            var_list,
            config.logs_path,
            summary_dict={
                'loss':loss_saver,
                'training_accuracy':training_accuracy_saver,
                'validation_accuracy':validation_accuracy_saver,
                'learning_rate':lr},
            restore=None)
    
    # Time measurements
    init_time = time()
    # Training --> use the configuration file
    print('Training...')
    with tf.compat.v1.Session() as sess:
        # Initialize
        init.run()
        # Restore former parameters
        if config.restore:
            print('Restoring weight stored in {}'.format(config.restore))
            saver.restore(sess, config.restore)

        # Training ...
        cur_img = config.start_img
        max_valid_acc = 0
        validation_accuracy_ = 0
        while cur_img < config.end_img:
            # images_bytes,_ = define_triplets_batch(training_filenames,training_labels,nbof_triplet=config.minibatch_size)
            images_bytes, lab_ = define_batch(training_filenames,training_labels,config.minibatch_size)
            images_ = [sess.run(image_decoded, feed_dict={image_jpeg:im}) for im in images_bytes]
            # Change label range:
            _, counts = np.unique(lab_, return_counts=True)
            lab_ = np.concatenate([[i]*j for i,j in enumerate(counts)])
            # images_,_ = online_adaptive_hard_image_generator(
            #     training_filenames, training_labels,
            #     sess,
            #     image_jpeg, images, training,
            #     image_decoded, emb,
            #     validation_accuracy_, config.minibatch_size, nbof_subclasses=40
            # )
            feed_dict = {
                images:   images_,
                label_inputs: lab_,
                training: True,
                lr:       training_schedule(cur_img)}

            loss_, training_accuracy_, _, _ = sess.run([loss, training_accuracy, training_op, extra_training_op], feed_dict=feed_dict)

            # Validation
            if cur_img % config.img_per_val == 0:
                pred_validation = np.empty((0,config.emb_size))
                labels_test = np.empty((0))
                for i in range(0,len(validation_labels),config.minibatch_size):
                    # images_ = load_images(validation_filenames[i:i+config.minibatch_size])
                    images_ = [sess.run(image_decoded, feed_dict={image_jpeg:im}) for im in validation_filenames[i:i+config.minibatch_size]]
                    feed_dict = {
                        images:   images_,
                        training: False}
                    pred_validation = np.append(pred_validation, sess.run(emb, feed_dict), axis=0)
                    labels_test = np.append(labels_test,validation_labels[i:i+config.minibatch_size])
                _, _, acc, val, _, far=evaluate.evaluate(pred_validation, labels_test[::2])
                validation_accuracy_ = np.mean(acc)
                feed_dict_saver = {
                    loss_saver: loss_,
                    training_accuracy_saver: training_accuracy_,
                    validation_accuracy_saver: validation_accuracy_,
                    lr:training_schedule(cur_img)}
                saver.save_summary(sess, feed_dict_saver, cur_img)
                # Display information
                graph_time  = time() - init_time
                hours       = int(graph_time // 3600)
                minutes     = int(graph_time // 60) - hours*60
                secondes    = graph_time - minutes * 60 - hours * 3600
                print('{:4d} kimgs, {}h {:2d}m {:2.1f}s, lr:{:0.6f}, training: {:2.2f}%, validation: {:2.2f}%'.format(
                    cur_img//1000,
                    hours, minutes,
                    secondes,
                    training_schedule(cur_img),
                    training_accuracy_*100,
                    validation_accuracy_*100))
            # Save logs
            if cur_img % config.img_per_summary == 0:
                feed_dict_saver = {
                    loss_saver: loss_,
                    training_accuracy_saver: training_accuracy_,
                    validation_accuracy_saver: validation_accuracy_,
                    lr:training_schedule(cur_img)}
                saver.save_summary(sess, feed_dict_saver, cur_img)
            # Save model
            # if cur_img % config.img_per_save == 0:
            if max_valid_acc < validation_accuracy_:
                saver.save_model(sess, cur_img//1000)
                max_valid_acc = validation_accuracy_
            # Update current image
            cur_img += config.minibatch_size
        # Final Saving
        saver.save_model(sess, 'final')
        saver.close_summary()
    print('Done: training')


def train_recent_from_npy():
    """
    Train the classification or the recognition network
    mode is in ['classification', 'recognition']
    """
    # Prepare the dataset
    training_filenames, training_labels_bytes = np.load(config.data_path)
    training_labels = np.array([int(tlb.decode('ascii')) for tlb in training_labels_bytes]) # Convert labels type: from bytes to int
    validation_filenames, validation_labels_bytes = np.load(config.data_test_path)
    validation_labels = np.array([int(vlb.decode('ascii')) for vlb in validation_labels_bytes]) # Convert labels type: from bytes to int

    # Graph definition: inputs, model, loss, optimizer, initializer
    print('Graph building...')
    # Small graph for decoding jpegs
    with tf.compat.v1.variable_scope('DecodeJPEG'):
        image_jpeg  = tf.compat.v1.placeholder(tf.string,shape=None,name='images_raw')
        image_decoded = tf.io.decode_jpeg(image_jpeg)
    # Inputs
    images       = tf.compat.v1.placeholder(tf.float32,shape=[None,config.image_size,config.image_size,3],name='image_inputs')
    label_inputs = tf.compat.v1.placeholder(tf.int64, shape=[None,], name='label_inputs')
    training     = tf.placeholder(tf.bool, shape=[], name='training')
    lr           = tf.compat.v1.placeholder(tf.float32, shape=[], name='learning_rate')
    # Augmentation
    image_inputs = images/127.5 - 1
    aug = lambda: dataset.augment_image(
        image_inputs,
        config.minibatch_size,
        use_horizontal_flip=True,
        rotation_rate   =0.3,
        translation_rate=0.2,
        cutout_size     =25,
        crop_pixels     =10)
    image_inputs = tf.cond(training, aug, lambda: image_inputs)

    with tf.variable_scope('Network'):
        # output = deep_cnn_v1(image_inputs_ph, training, config.emb_size, regularizer_rate)
        output = wideresnet(image_inputs,training,config.emb_size,regularizer_rate=config.regularizer_rate,fmaps=[80,160,320,640],nbof_unit=[4,4,4,4],strides=[2,2,2,2],dropouts=[0.,0.,0.,0.])
        # output = deep_cnn_v1(image_inputs, training, config.emb_size, regularizer_rate=config.regularizer_rate, fmaps=[16,32,64,128])
        # output = insightface_resnet(
        #     inputs,
        #     training,
        #     emb_size,
        #     regularizer_rate=regularizer_rate,
        #     dropout_rate=0.,
        #     fmaps       = [16,16,32,64,128],
        #     nbof_unit   = [1,1,1,1,1],
        #     strides     = [2,2,2,2,2])
        emb = tf.nn.l2_normalize(output, axis=1)
        
    # Loss network
    with tf.variable_scope('LossLayers'):
        # net_logit_layer = layers.dense(emb, config.nbof_labels)
        # net_logit_layer = losses.cosineface_losses(emb, label_inputs, config.nbof_labels, config.regularizer_rate)
        net_logit_layer = losses.cosineface_losses(emb, label_inputs, config.minibatch_size, config.regularizer_rate)
        net_loss_layer = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net_logit_layer, labels=label_inputs))

    input_emb = tf.compat.v1.placeholder(tf.float32, shape=[None, config.emb_size], name='input_emb')
    with tf.variable_scope('LossLayers', reuse=True):
        # logit_layer = layers.dense(input_emb, config.nbof_labels)
        # logit_layer = losses.cosineface_losses(input_emb, label_inputs, config.nbof_labels, config.regularizer_rate)
        logit_layer = losses.cosineface_losses(input_emb, label_inputs, config.minibatch_size, config.regularizer_rate)
        loss_layer = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_layer, labels=label_inputs))
    # Optimizer
    optimizer_net       = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, **config.optimzer_kwargs)
    optimizer_loss      = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
    # optimizer_loss      = tf.compat.v1.train.AdamOptimizer(learning_rate=lr*10, **config.optimzer_kwargs)
    var_list_net        = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Network')
    var_list_loss       = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='LossLayers')
    assert len(var_list_net)>0
    # Training operations
    loss_net = tf.compat.v1.placeholder(tf.float32, shape=[], name='loss')
    training_op_net     = optimizer_net.minimize(net_loss_layer, var_list=var_list_net, name='training_op_net')
    training_op_loss    = optimizer_loss.minimize(loss_layer, var_list=var_list_loss, name='training_op_loss')
    extra_training_op = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS) # For batch normalization
    # Initializer
    init = tf.compat.v1.global_variables_initializer()
    print('Done: graph built.')

    # Saver
    with tf.compat.v1.variable_scope('Saver'):
        train_acc = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(tf.nn.softmax(net_logit_layer), 1),label_inputs), tf.float32))
        training_accuracy = tf.compat.v1.placeholder(tf.float32, shape=[], name='training_accuracy_placeholder')
        validation_accuracy = tf.compat.v1.placeholder(tf.float32, shape=[], name='validation_accuracy_placeholder')
        saver = train.Saver(var_list_net,config.logs_path, summary_dict={'loss':loss_net, 'training_accuracy':training_accuracy, 'validation_accuracy':validation_accuracy, 'learning_rate':lr}, restore=None)
    
    # Time measurements
    init_time = time()
    # Training --> use the configuration file
    print('Training...')
    with tf.compat.v1.Session() as sess:
        # Initialize
        init.run()
        # Restore former parameters
        if config.restore:
            print('Restoring weight stored in {}'.format(config.restore))
            saver.restore(sess, config.restore)

        # Training ...
        cur_img = config.start_img
        max_valid_acc = 0
        validation_accuracy_ = 0
        while cur_img < config.end_img:

            # Inputs
            img_, lab_ = define_batch(training_filenames,training_labels,config.minibatch_size)
            images_ = [sess.run(image_decoded, feed_dict={image_jpeg:im}) for im in img_]
            # Change label range:
            _, counts = np.unique(lab_, return_counts=True)
            lab_ = np.concatenate([[i]*j for i,j in enumerate(counts)])
            # Embeddings
            emb_ = sess.run(emb, feed_dict={images:images_, label_inputs:lab_, training:True})
            # Training operation
            for _ in range(100): # Overtrain the last layer
                sess.run(training_op_loss, feed_dict={input_emb:emb_, label_inputs:lab_, lr:training_schedule(cur_img)})
            # Train the network
            loss_net_,training_accuracy_,_,_=sess.run([net_loss_layer, train_acc, training_op_net,extra_training_op], feed_dict={images:images_, label_inputs:lab_, training:True, lr:training_schedule(cur_img)})
  
            # Validation
            if cur_img % config.img_per_val == 0:
                pred_validation = np.empty((0,config.emb_size))
                labels_test = np.empty((0))
                for i in range(0,len(validation_labels),config.minibatch_size):
                    # images_ = load_images(validation_filenames[i:i+config.minibatch_size])
                    images_ = [sess.run(image_decoded, feed_dict={image_jpeg:im}) for im in validation_filenames[i:i+config.minibatch_size]]
                    feed_dict = {
                        images:   images_,
                        training: False}
                    pred_validation = np.append(pred_validation, sess.run(emb, feed_dict), axis=0)
                    labels_test = np.append(labels_test,validation_labels[i:i+config.minibatch_size])
                _, _, acc, val, _, far=evaluate.evaluate(pred_validation, labels_test[::2])
                validation_accuracy_ = np.mean(acc)
                # Display information
                graph_time  = time() - init_time
                hours       = int(graph_time // 3600)
                minutes     = int(graph_time // 60) - hours*60
                secondes    = graph_time - minutes * 60 - hours * 3600
                print('{:4d} kimgs, {}h {:2d}m {:2.1f}s, lr:{:0.6f}, training: {:2.2f}%, validation: {:2.2f}%'.format(
                    cur_img//1000,
                    hours, minutes,
                    secondes,
                    training_schedule(cur_img),
                    training_accuracy_*100,
                    validation_accuracy_*100))
            # Save logs
            if cur_img % config.img_per_summary == 0:
                feed_dict_saver = {
                    images:images_,
                    label_inputs:lab_,
                    validation_accuracy:validation_accuracy_,
                    training_accuracy:training_accuracy_,
                    training:False,
                    loss_net:loss_net_,
                    lr:training_schedule(cur_img)}
                saver.save_summary(sess, feed_dict_saver, cur_img)
            # Save model
            # if cur_img % config.img_per_save == 0:
            if max_valid_acc < validation_accuracy_:
                saver.save_model(sess, cur_img//1000)
                max_valid_acc = validation_accuracy_
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
