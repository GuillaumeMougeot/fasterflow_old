import tensorflow as tf
from functools import partial
import os
from shutil import copyfile
from datetime import datetime
import skimage as sk
import numpy as np
from time import time

from fasterflow import train
# from fasterflow import dataset
from fasterflow import utils
from fasterflow import evaluate
import data as dataset
import config_pro_gan as config
import network_pro_gan as network 

from midi_utils import keyed_pianoroll2midi

inputs = getattr(dataset, config.data_initilize)(**config.data_initilize_kwargs)
# Define the minibatch selector
select_minibatch = partial(getattr(dataset, config.data_selector), inputs)

for j in range(0,10000,25):
    out = select_minibatch(crt_img=j, res=256, minibatch_size=25)
    # out = utils.downscale2d_np(out)/2 +0.5
    print(out.shape)
    for i in range(len(out)):
        im = out[i]*255
        im = im.astype(np.uint8)
        sk.io.imsave('logs/images/{}.jpg'.format(j+i), im)
    
    for i in range(len(out)):
        keyed_pianoroll2midi(out[i]*100, file_path='logs/musics/{}.mid'.format(j+i))