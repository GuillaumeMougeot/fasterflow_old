import tensorflow as tf
from functools import partial
import os
from shutil import copyfile
from datetime import datetime
import skimage as sk
import numpy as np
from time import time

from fasterflow import utils

#----------------------------------------------------------------------------
# Saver wrapper

class Saver:
    def __init__(self,
        var_list,           # List of variables to save in the saver
        logs_path,          # Path of the logs folder
        summary_dict={},    # List of variables to save in the summary
        restore=None):      # Restoring folder
        # Saver
        self.saver = tf.compat.v1.train.Saver(var_list=var_list)

        # Create summary and logs folders
        if restore==None:
            now = datetime.now().strftime('%Y%m%d%H%M%S')
            self.logdir = '{}-run_{}/'.format(logs_path,now)
            os.makedirs(self.logdir+'models/')
            os.makedirs(self.logdir+'images/')
        else:
            self.logdir = restore[:restore.rfind('models/')]

        # Save the config.py file
        # copyfile('config.py', self.logdir+'config.py')

        # Summary
        self.summary_vars = []
        for key, value in summary_dict.items():
            self.summary_vars += [tf.compat.v1.summary.scalar(key, tf.compat.v1.math.reduce_sum(value))]
        self.summary_writer = tf.compat.v1.summary.FileWriter(self.logdir+'tensorboard/', tf.compat.v1.get_default_graph())

    def restore(self, sess, save_path): # TODO: improve restore
        self.saver.restore(sess, save_path)

    def save_summary(self, sess, feed_dict, cur_img):
        summary_str = sess.run(self.summary_vars,feed_dict=feed_dict)
        for summary in summary_str:
            self.summary_writer.add_summary(summary, cur_img)

    def save_model(self, sess, cur_img):
        self.saver.save(sess, '{}models/model-{}.ckpt'.format(self.logdir,cur_img))
    
    def save_images(self, images, cur_img):
        # utils.save_images('{}images/{}.jpg'.format(self.logdir,cur_img), images, images.shape[2])
        # gw = np.clip(1080 // images.shape[3], 2, 32)
        # gh = np.clip(1080 // images.shape[2], 2, 32)
        images = np.transpose(images,axes=(0,3,1,2))
        utils.save_image_grid(images, '{}images/{}.jpg'.format(self.logdir,cur_img), drange=[0,1], grid_size=None)
    
    def close_summary(self):
        self.summary_writer.close()

#----------------------------------------------------------------------------
