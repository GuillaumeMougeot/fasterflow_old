import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from fasterflow import prepare_data
from fasterflow import train
from fasterflow import dataset
from fasterflow import network

import config_classification as config
import network_classification

# Load data
validation_dataset = getattr(dataset, config.data_test_initilize)(minibatch_size=config.minibatch_size, repeat=False, **config.data_test_initilize_kwargs)
validation_iterator = validation_dataset.make_one_shot_iterator()
valid_data = validation_iterator.get_next()

images, labels = prepare_data.prepare_cifar10('../../data/cifar10/')
images=images.astype(np.uint8)
labels=labels.astype(np.int64)
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
n = 128

# plt.imshow(images[:n])
# plt.show()
# Load graph

# Network parameters
image_inputs = tf.placeholder(tf.float32, shape=[None,config.image_size,config.image_size,3], name='image_inputs')
image_normed = image_inputs/127.5
image_normed = image_normed - 1
label_inputs = tf.placeholder(tf.int64, shape=[None], name='label_inputs')
nbof_labels = config.nbof_labels
emb_size = config.latent_size
training = tf.placeholder(tf.bool, shape=[], name='training')
# Network
net = network.Network('classification', network_classification.v3, image_normed, labels, emb_size, training, nbof_labels)
logit = net(image_normed)
var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='classification')
pred = tf.nn.softmax(logit)
pred = tf.argmax(pred, axis=1)
# Load network
# saver_path = 'logs/Cifar10-classification_v1-run_20191026172524/models/'
# saver_path = 'logs/Cifar10-classification_v1-run_20191028120938/models/' # without batch norm (1568)
saver_path = 'logs/Cifar10-classification_v1-run_20191028144237/models/' # with custom norm custom
# saver_path = 'logs/Cifar10-classification_v1-run_20191028151112/models/' # with keras batch norm


saver = tf.compat.v1.train.Saver(var_list=var_list)
# Run session on samples
with tf.Session() as sess:
    saver.restore(sess, saver_path+'model-336.ckpt')
    
    feed_dict = {
        image_inputs:images[-n:].astype(np.float32),
        label_inputs:labels[-n:].astype(np.float32),
        training: False
    }
    pred_ = sess.run(pred, feed_dict)
    out = [label_names[i] for i in pred_]
    real_out = [label_names[i] for i in labels[-n:]]
    for i in range(len(out)):
        print('{},{}'.format(real_out[i],out[i]))
    # pred_ = sess.run(pred, {''})

    # Validation set
    pred_validation = np.empty((0))
    labels_test = np.empty((0))
    while True:
        try:
            imgs, labs = sess.run(valid_data)
            labs = labs.flatten()
            feed_dict = {
                image_inputs: imgs.astype(np.float32),
                label_inputs: labs.astype(np.float32),
                training: False
            }
            pred_, labs_ = sess.run([pred, label_inputs], feed_dict=feed_dict)
            pred_validation = np.append(pred_validation, pred_, axis=0)
            labels_test = np.append(labels_test,labs_)
        except tf.errors.OutOfRangeError:
            break
    valid_acc = np.mean(np.equal(pred_validation, labels_test))
    # Display information
    print(valid_acc)            
    