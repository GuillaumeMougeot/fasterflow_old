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

#----------------------------------------------------------------------------
# Detection

def detection(
    inputs,                 # Input images
    labels,                 # Input labels
    nbof_labels,            # Number of different labels (number of identities)
    training,               # Use dropout or not? (Training or not training?)
    regularizer_rate=0):    # Use weight regularization?
    net = network.Network('classification', wideresnet_se, inputs, training, nbof_labels, regularizer_rate=regularizer_rate)
    logit = net(inputs)
    # with tf.compat.v1.variable_scope('classification'):
    #     emb = deep_cnn_v1(inputs, training, 128, regularizer_rate=regularizer_rate)
    #     emb = tf.nn.l2_normalize(emb)
    #     logit = losses.cosineface_losses(emb, labels, nbof_labels, regularizer_rate=regularizer_rate)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_losses = tf.compat.v1.add_n(reg_losses, name='reg_loss')
    return logit, loss, reg_losses

#----------------------------------------------------------------------------
# Classification

def classification(
    inputs,                 # Input images
    labels,                 # Input labels
    nbof_labels,            # Number of different labels (number of identities)
    training,               # Use dropout or not? (Training or not training?)
    regularizer_rate=0):    # Use weight regularization?
    net = network.Network('classification', wideresnet_se, inputs, training, nbof_labels, regularizer_rate=regularizer_rate)
    logit = net(inputs)
    # with tf.compat.v1.variable_scope('classification'):
    #     emb = deep_cnn_v1(inputs, training, 128, regularizer_rate=regularizer_rate)
    #     emb = tf.nn.l2_normalize(emb)
    #     logit = losses.cosineface_losses(emb, labels, nbof_labels, regularizer_rate=regularizer_rate)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_losses = tf.compat.v1.add_n(reg_losses, name='reg_loss')
    return logit, loss, reg_losses

#----------------------------------------------------------------------------
# Recognition

def recognition(
    inputs,                 # Input images
    labels,                 # Input labels
    emb_size,               # Size of the embedding vector
    nbof_labels,            # Number of different labels (number of identities)
    training,               # Use dropout or not? (Training or not training?)
    use_adaptive_loss=False,
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
        #     fmaps       = [16,16,32,64,128,512],
        #     nbof_unit   = [1,1,1,1,1,1],
        #     strides     = [2,2,2,2,2,2])
        output = insightface_resnet(
            inputs,
            training,
            emb_size,
            regularizer_rate=regularizer_rate,
            dropout_rate=0.,
            fmaps       = [64,128,256,512,1024],
            nbof_unit   = [2,3,5,8,3],
            strides     = [2,2,2,2,2])
        # output = wideresnet(
        #     inputs,
        #     training,
        #     emb_size,
        #     regularizer_rate=regularizer_rate,
        #     fmaps       =[80,160,320,640],
        #     nbof_unit   =[4,4,4,4],
        #     strides     =[2,2,2,2],
        #     # dropouts    =[0.1,0.1,0.2,0.3])
        #     dropouts    =[0.,0.,0.,0.])
        emb = tf.nn.l2_normalize(output, axis=1)
        logit = losses.cosineface_losses(emb, labels, nbof_labels, regularizer_rate=regularizer_rate)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))
        # loss = losses.adaptive_triplet_loss(emb, labels, nbof_labels) if use_adaptive_loss else losses.general_triplet_loss(emb, labels, nbof_labels)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_losses = tf.compat.v1.add_n(reg_losses, name='reg_loss')
    return emb, logit, loss, reg_losses
    # return emb, loss, reg_losses

#----------------------------------------------------------------------------
