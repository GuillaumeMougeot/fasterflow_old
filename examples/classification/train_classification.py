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

import config_classification as config
import network_classification as network

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

def train_classification(mode='classification', use_adaptive_loss=False):
    """
    Train the classification or the recognition network
    mode is in ['classification', 'recognition']
    """
    assert mode in ['classification', 'recognition']
    # Graph definition: inputs, model, loss, optimizer, initializer
    print('Graph building...')
    with tf.compat.v1.variable_scope('Dataset'):
        # Init dataset
        training_dataset = getattr(dataset, config.data_initilize)(minibatch_size=config.minibatch_size, shuffle=True, **config.data_initilize_kwargs)
        validation_dataset = getattr(dataset, config.data_test_initilize)(minibatch_size=config.minibatch_size, shuffle=False, repeat=False, **config.data_test_initilize_kwargs)
    # Inputs
    handle = tf.placeholder(tf.string, shape=[], name='handle_input')
    iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
    # image_ins, label_ins = iterator.get_next()
    image_inputs, label_inputs = iterator.get_next()
    image_inputs.set_shape([None,config.image_size,config.image_size,3])
    label_inputs.set_shape([None,1])
    label_inputs = tf.reshape(label_inputs, [-1])
    label_inputs = tf.cast(label_inputs, tf.int64)
    image_inputs = tf.cast(image_inputs, dtype=tf.float32)/127.5 - 1
    # Network parameters
    nbof_labels = config.nbof_labels
    training = tf.placeholder(tf.bool, shape=[], name='training')
    regularizer_rate=config.regularizer_rate
    # Augmentation
    # if mode=='classification':
    # aug = lambda: dataset.augment_image(
    #     image_inputs,
    #     config.minibatch_size,
    #     use_horizontal_flip=(mode=='classification'),
    #     rotation_rate   =0.3,
    #     translation_rate=0.1,
    #     cutout_size     =25,
    #     crop_pixels     =10)
    # image_inputs = tf.cond(training, aug, lambda: image_inputs)
    # Network
    if mode=='classification':
        logit,loss,reg_loss = network.classification(image_inputs, label_inputs, nbof_labels, training, regularizer_rate)
    else:
        emb_size = config.emb_size
        emb,logit,loss,reg_loss = network.recognition(image_inputs, label_inputs, emb_size, nbof_labels, training, use_adaptive_loss=False, regularizer_rate=regularizer_rate)
        # emb,loss,reg_loss = network.recognition(image_inputs, label_inputs, emb_size, nbof_labels, training, use_adaptive_loss=use_adaptive_loss, regularizer_rate=regularizer_rate)
    # Optimizer
    # optimizer           = tf.compat.v1.train.MomentumOptimizer(learning_rate=config.learning_rate, momentum=0.9, use_nesterov=True)
    lr                  = tf.compat.v1.placeholder(tf.float32, shape=[], name='lr')
    optimizer           = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, **config.optimzer_kwargs)
    var_list            = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=config.network)
    assert len(var_list)>0
    # Training operations
    training_op         = optimizer.minimize(loss+reg_loss, var_list=var_list, name='training_op')
    extra_training_op   = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS) # For batch normalization
    # Initializer
    init                = tf.compat.v1.global_variables_initializer()
    print('Done: graph built.')

    # Define the minibatch selector
    with tf.compat.v1.variable_scope('Dataset'):
        training_iterator = training_dataset.make_one_shot_iterator()
        validation_iterator = validation_dataset.make_initializable_iterator()

    # Saver
    with tf.compat.v1.variable_scope('Saver'):
        if mode == 'classification':
            pred = tf.nn.softmax(logit)
            training_output = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), label_inputs), dtype=tf.float32))
        else:
            pred = tf.nn.softmax(logit)
            training_output = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), label_inputs), dtype=tf.float32))
            # training_output = losses.general_triplet_accuracy(emb, label_inputs, nbof_labels)
        training_accuracy = tf.compat.v1.placeholder(tf.float32, shape=[], name='training_accuracy_placeholder')
        validation_accuracy = tf.compat.v1.placeholder(tf.float32, shape=[], name='validation_accuracy_placeholder')
        saver = train.Saver(var_list,config.logs_path, summary_dict={'loss':loss, 'reg_loss':reg_loss, 'training_accuracy':training_accuracy, 'validation_accuracy':validation_accuracy, 'lr':lr}, restore=None)
    
    # Time measurements
    init_time = time()
    train_acc = 0
    img_per_train_acc = 0
    saved_train_acc = 0
    valid_acc = 0
    # Training --> use the configuration file
    print('Training...')
    with tf.compat.v1.Session() as sess:
        # Initialize
        init.run()
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())
        # Restore former parameters
        if config.restore:
            print('Restoring weight stored in {}'.format(config.restore))
            saver.restore(sess, config.restore)

        # Training ...
        cur_img = config.start_img
        while cur_img < config.end_img:
            # Training operation
            feed_dict = {handle: training_handle, lr:training_schedule(cur_img), training:True}

            # feed_dict = {handle: validation_handle, lr:training_schedule(cur_img), training:False}
            # import matplotlib.pyplot as plt
            # sess.run(validation_iterator.initializer)
            # for i in range(1):
            #     imgs, labs = sess.run([image_inputs,label_inputs],feed_dict=feed_dict)
            # print(labs)
            # fig = plt.figure()
            # fig.set_size_inches(1, 1, forward=False)
            # ax = plt.Axes(fig, [0., 0., 1., 1.])
            # ax.set_axis_off()
            # fig.add_axes(ax)
            # ax.imshow(imgs[1]/2+0.5)
            # n = int(np.sqrt(len(imgs)))
            # for i in range(n):
            #     for j in range(n):
            #         plt.subplot(n,n,i*n+j+1)
            #         plt.imshow(imgs[i*n+j]/2 + 0.5)
            # plt.show()
            # return 0

            _,_, train_out = sess.run([training_op, extra_training_op, training_output], feed_dict=feed_dict)
            
            # Save accuracy
            train_acc += train_out
            img_per_train_acc += 1

            # Validation
            if cur_img % config.img_per_val == 0:
                pred_validation = np.empty((0,nbof_labels) if mode=='classification' else (0,emb_size))
                labels_test = np.empty((0))
                sess.run(validation_iterator.initializer)
                while True:
                    try:
                        pred_, labs = sess.run([pred, label_inputs] if mode=='classification' else [emb, label_inputs],
                            feed_dict={handle: validation_handle, training:False})
                        pred_validation = np.append(pred_validation, pred_, axis=0)
                        labels_test = np.append(labels_test,labs)
                    except tf.errors.OutOfRangeError:
                        break
                if mode=='classification':
                    valid_acc = np.mean(np.equal(np.argmax(pred_validation, axis=1), labels_test))
                else:
                    _, _, acc, val, _, far=evaluate.evaluate(pred_validation, labels_test[::2])
                    valid_acc = np.mean(acc)
                feed_dict_saver = {handle: training_handle, validation_accuracy:valid_acc, training_accuracy:saved_train_acc, training:False, lr:training_schedule(cur_img)}
                saver.save_summary(sess, feed_dict_saver, cur_img)
                # Display information
                graph_time  = time() - init_time
                hours       = int(graph_time // 3600)
                minutes     = int(graph_time // 60) - hours*60
                secondes    = graph_time - minutes * 60 - hours * 3600
                print('{:4d} kimgs, {}h {:2d}m {:2.1f}s, lr:{:0.6f}, training: {:2.2f}%, validation: {:2.2f}%'.format(cur_img//1000, hours, minutes, secondes, training_schedule(cur_img), saved_train_acc*100, valid_acc*100))
            # Save logs
            if cur_img % config.img_per_summary == 0:
                assert img_per_train_acc!=0
                train_acc /= img_per_train_acc
                feed_dict_saver = {handle: training_handle, validation_accuracy:valid_acc, training_accuracy:train_acc, training:False, lr:training_schedule(cur_img)}
                saver.save_summary(sess, feed_dict_saver, cur_img)
                saved_train_acc = train_acc
                train_acc = 0
                img_per_train_acc = 0
            # Save model
            if cur_img % config.img_per_save == 0:
                saver.save_model(sess, cur_img//1000)
            # Update current image
            cur_img += config.minibatch_size
        # Final Saving
        saver.save_model(sess, 'final')
        saver.close_summary()
        sess.close()
    print('Done: training')

#----------------------------------------------------------------------------
# Test

if __name__=='__main__':
    # train_classification(mode='recognition', use_adaptive_loss=False)
    train_classification(mode='recognition', use_adaptive_loss=True)
    # exec(config.training_function)

#----------------------------------------------------------------------------
