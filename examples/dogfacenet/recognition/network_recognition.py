import tensorflow as tf 
import skimage as sk 
import matplotlib.pyplot as plt
import numpy as np
import os

from fasterflow import layers
from fasterflow import utils
from fasterflow import losses
from fasterflow import network

from fasterflow.network_applications.wideresnet import wideresnet, wideresnet_se
from fasterflow.network_applications.insightface_resnet import insightface_resnet
from fasterflow.network_applications.keras_resnet import keras_resnet
from fasterflow.network_applications.deep_cnn import deep_cnn_v1

import triplet_loss

#----------------------------------------------------------------------------
# Recognition

def recognition(
    inputs,                 # Input images
    emb_size,               # Size of the embedding vector
    training,               # Use dropout or not? (Training or not training?)
    regularizer_rate=0):    # Use weight regularization?
    with tf.compat.v1.variable_scope('recognition'):
        output = deep_cnn_v1(inputs, training, emb_size, regularizer_rate=regularizer_rate, fmaps=[16,32,64,128])
        # output = insightface_resnet(
        #     inputs,
        #     training,
        #     emb_size,
        #     regularizer_rate=regularizer_rate,
        #     dropout_rate=0.,
        #     fmaps       = [16,16,32,64,128],
        #     nbof_unit   = [1,1,1,1,1],
        #     strides     = [2,2,2,2,2])
        # output = wideresnet(inputs,training,emb_size,regularizer_rate=regularizer_rate,fmaps=[80,160,320,640],nbof_unit=[4,4,4,4],strides=[2,2,2,2],dropouts=[0.,0.,0.,0.])
        emb = tf.nn.l2_normalize(output, axis=1)
    loss = losses.triplet_loss(emb)
    # loss = losses.micheal_accuracy(emb, labels, nbof_labels)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_losses = tf.compat.v1.add_n(reg_losses, name='reg_loss')
    return emb, loss, reg_losses

def recognition_micheal(
    inputs,                 # Input images
    emb_size,               # Size of the embedding vector
    labels,
    nbof_labels,
    training,               # Use dropout or not? (Training or not training?)
    regularizer_rate=0):    # Use weight regularization?
    with tf.compat.v1.variable_scope('recognition'):
        # output = deep_cnn_v1(
        #     inputs,
        #     training,
        #     emb_size,
        #     regularizer_rate=regularizer_rate,
        #     fmaps=[16,32,64,128])
        # output = insightface_resnet(
        #     inputs,
        #     training,
        #     emb_size,
        #     regularizer_rate=regularizer_rate,
        #     dropout_rate=0.,
        #     fmaps       = [16,16,32,64,128],
        #     nbof_unit   = [1,1,1,1,1],
        #     strides     = [2,2,2,2,2])
        output = wideresnet(
            inputs,
            training,
            emb_size,
            regularizer_rate=regularizer_rate,
            fmaps       =[80,160,320,640],
            nbof_unit   =[4,4,4,4],
            strides     =[2,2,2,2],
            dropouts    =[0.,0.,0.,0.])
        emb = tf.nn.l2_normalize(output, axis=1)
    # loss = losses.triplet_loss(emb)
    # loss = losses.general_triplet_loss(emb, labels, nbof_labels)
    loss = losses.focal_general_triplet_loss(emb, labels, nbof_labels)
    # loss = triplet_loss.batch_hard_triplet_loss(labels, emb, 0.2, True)
    # loss,_ = triplet_loss.batch_all_triplet_loss(labels, emb, 0.2, True)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_losses = tf.compat.v1.add_n(reg_losses, name='reg_loss')
    return emb, loss, reg_losses

#----------------------------------------------------------------------------
