import os
import numpy as np
import skimage as sk 
import matplotlib.pyplot as plt
import tensorflow as tf
from functools import partial

from fasterflow import utils
from fasterflow import layers

#----------------------------------------------------------------------------
# Numpy minibatch selectors

# GAN related functions
def select_minibatch_gan_cifar10(images, crt_img, res, minibatch_size):
    # Choose the appropriate indices
    start_idx = crt_img % len(images)
    end_idx = start_idx + minibatch_size
    if end_idx >= len(images):
        choice_idx = np.arange(start_idx,len(images))
        choice_idx = np.append(choice_idx,np.arange(0,end_idx-len(images)))
    else:
        choice_idx = np.arange(start_idx,end_idx)
    # Resize image
    factor = images.shape[2]//res
    # If the size of the target images is the same as the input images, no need to resize
    if factor==1:
        return 2*images[choice_idx]-1
    else:
        # First downscale the images
        x = 2*utils.downscale2d_np(images[choice_idx],factor=factor)-1
        # Then upscale again to match the network expectation
        x = utils.upscale2d_np(x, factor=factor)
        return x

# Recognition related functions
# Images have to be [250,250,3]
def select_minibatch_recognition(filenames, labels, crt_img, minibatch_size):
    # Choose the appropriate indices
    start_idx = crt_img % len(filenames)
    end_idx = start_idx + minibatch_size
    if end_idx >= len(filenames):
        choice_idx = np.arange(start_idx,len(filenames))
        choice_idx = np.append(choice_idx,np.arange(0,end_idx-len(filenames)))
    else:
        choice_idx = np.arange(start_idx,end_idx)
    # Select filenames and labels
    selected_filenames = filenames[choice_idx]
    selected_labels = np.expand_dims(labels[choice_idx], axis=1)
    # Load images
    images = np.empty((minibatch_size,250,250,3))
    labels = np.empty((minibatch_size))
    readable = 0
    for i in range(len(selected_filenames)):
        image = sk.io.imread(selected_filenames[i]) # 250x250x3
        image = np.expand_dims(image,axis=0)
        if len(image.shape)==4: # If it can read the file
            # image = utils.downscale2d_np(image) # 125x125x3
            images[readable] = image[0]
            labels[readable]=selected_labels[i]
            readable += 1
    images = images[:readable]/127.5 - 1
    labels = labels[:readable]
    assert len(images)==len(labels), '[Error] Cannot output {} images and {} labels'.format(len(images),len(selected_labels))
    assert len(labels)>0, '[Error] Not enough readable files.'
    return images, labels

#----------------------------------------------------------------------------
# Tensorflow (from tfrecord file) minibatch selectors

def from_tfrecord_parse_function(record, read_from_bytes=False):
    features = tf.compat.v1.parse_single_example(record, features={
        'shape': tf.compat.v1.FixedLenFeature([3], tf.int64),
        'image': tf.compat.v1.FixedLenFeature([], tf.string),
        'label': tf.compat.v1.FixedLenFeature([1], tf.int64)})
    data = tf.compat.v1.io.decode_jpeg(features['image']) if read_from_bytes else tf.compat.v1.decode_raw(features['image'], tf.uint8)
    return tf.compat.v1.reshape(data, features['shape']), features['label']

def from_tfrecord_read_images_labels(
    data_path,              # Tfrecord file path
    minibatch_size,         # Size of the minibatch
    shuffle=False,          # Suffle the data?
    repeat=True,            # Infinity loop over the data
    read_from_bytes=False): # Were images stored in jpeg format?
    """
    Returns a tf.data.Dataset object defined by tfrecord file.
    """
    assert os.path.isfile(data_path), '[Error] {} is not file.'.format(data_path)
    dataset = tf.compat.v1.data.TFRecordDataset(data_path, compression_type='')
    parse = partial(from_tfrecord_parse_function, read_from_bytes=read_from_bytes)
    dataset = dataset.map(parse)
    if shuffle: dataset = dataset.shuffle(buffer_size=10000)
    if repeat: dataset = dataset.repeat()
    dataset = dataset.batch(minibatch_size)
    return dataset

def from_filenames_labels_parse_function(filename, label):
    with tf.compat.v1.variable_scope('read_image'):
            image_raw = tf.compat.v1.io.read_file(filename)
            image = tf.compat.v1.io.decode_jpeg(image_raw)
            image = tf.cast(image, tf.float32)
    return image, label

def from_filenames_labels_read_images(
    filenames,      # Numpy array with the filenames
    labels,         # Numpy array of labels 
    minibatch_size, # Size of the minibatch
    shuffle=False,  # Should the dataset be shuffled?
    repeat=True):   # Should we repeat the dataset indefinetly?
    """
    Returns a tf.data.Dataset object defined by filenames and labels.
    """
    with tf.compat.v1.variable_scope('from_filenames_labels_read_images'):
        dataset_filenames = tf.compat.v1.data.Dataset.from_tensor_slices(filenames)
        dataset_labels = tf.compat.v1.data.Dataset.from_tensor_slices(labels)
        dataset = tf.compat.v1.data.Dataset.zip((dataset_filenames,dataset_labels))
        dataset = dataset.map(from_filenames_labels_parse_function)
        if shuffle: dataset = dataset.shuffle(buffer_size=9000)
        if repeat: dataset = dataset.repeat()
        dataset = dataset.batch(minibatch_size)
    return dataset

#----------------------------------------------------------------------------
# Data augmentation

# Image pre-processing
def augment_image(
    x,                          # Images inputs, size: [NHWC]
    minibatch_size,             # Minibatch size, needed for rotation, translation and cutout (has to be improved)
    use_horizontal_flip = True, # Use random horizontal flipping?
    rotation_rate       = 0.1,  # Random rotation angle range (in radian). If 0 then not used
    translation_rate    = 0.1,  # Random translation range (in percentage of the picture size). If 0 then not used
    cutout_size         = 16,   # Random cutout size (in pixels)
    crop_pixels         = 4):   # Random crop size (in pixels)
    if use_horizontal_flip:
        with tf.name_scope('LeftRigth'):
            x = tf.image.random_flip_left_right(x)
            # alea = tf.random.uniform([1])>0.5
            # x = tf.cond(alea, lambda: x, lambda: tf.reverse(x, axis=[2]))
    if rotation_rate != 0. or translation_rate > 0.:
        with tf.name_scope('RotateTranslate'):
            _,h,w,_ = x.shape.as_list()
            # Numpy constants
            j,k = np.expand_dims(np.arange(h),-1), np.expand_dims(np.arange(w),-1)
            j,k = np.expand_dims(np.tile(j,w),-1), np.expand_dims(np.tile(k,h).T,-1)
            h2,w2 = np.ones_like(j)*h/2, np.ones_like(k)*w/2
            h2w2 = np.concatenate([h2,w2],axis=-1)
            indices = np.concatenate([j,k], axis=-1)
            ## Center the indices
            indices = indices-h2w2
            # Add ones for homogeneous coordinates
            indices = np.concatenate([indices,np.ones_like(j)], axis=-1)
            # Tensorflow constant
            theta = tf.random.uniform([1])[0]*rotation_rate
            identity1 = tf.constant([[1,0],[0,1]], dtype=tf.float32)
            identity2 = tf.constant([[0,-1],[1,0]], dtype=tf.float32)
            ## Rotation matrix
            M = tf.cos(theta)*identity1+tf.sin(theta)*identity2
            # Add translation
            translation_range = int(translation_rate*x.shape.as_list()[2])
            trans = tf.random.uniform([2], minval=-translation_range, maxval=translation_range, dtype=tf.float32)
            M = tf.concat([M,[trans]],axis=0)
            ## Indices of the image before rotation
            indices_tf = tf.constant(indices, dtype=tf.float32)
            ## Center of the image
            hw = tf.constant(h2w2, dtype=tf.float32)
            # TODO: remove the minibatch_size dependency below
            n = minibatch_size
            # i is need for tf.scatter_nd function
            i = tf.tile(tf.reshape(tf.range(0,n), shape=[n,1,1,1]),[1,h,w,1])
            # Rotate the indices
            rot_idx = tf.matmul(indices_tf, M)
            # TODO: replace min(h,w) by the appropriate value (NOTE: it works normally for square images)
            rot_idx = tf.cast(tf.floor(tf.clip_by_value(rot_idx+hw,0,min(h,w))),tf.int32) 
            idx = tf.tile(tf.expand_dims(rot_idx,0),(n,1,1,1))
            idx = tf.concat([i,idx], axis=-1)
            # Apply the transformation
            x = tf.scatter_nd(idx, x, idx.shape)
    # Cutout: add a black square on the picture
    if cutout_size > 0:
        with tf.name_scope('Cutout'):
            cut_size = cutout_size
            _,h,w,_ = x.shape.as_list()
            coord = tf.random.uniform([2], minval=-cut_size//2, maxval=w-cut_size//2, dtype=tf.int64)
            # size = tf.random.uniform([2], minval=cut_size//2, maxval=cut_size, dtype=tf.int32)
            size = tf.constant([cut_size,cut_size], dtype=tf.int64)
            # Goal (but not possible in tensorflow): x[:,coord[0]:coord[0]+size[0],coord[1]:coord[1]+size[1],:] = 0
            j,k = np.expand_dims(np.arange(h),-1), np.expand_dims(np.arange(w),-1)
            j,k = np.expand_dims(np.tile(j,w),-1), np.expand_dims(np.tile(k,h).T,-1)
            indices = tf.constant(np.concatenate([j,k], axis=-1), dtype=tf.int64)
            cond = tf.logical_and(
                tf.logical_and(indices[:,:,0]>=coord[0],indices[:,:,0]<coord[0]+size[0]),
                tf.logical_and(indices[:,:,1]>=coord[1],indices[:,:,1]<coord[1]+size[1]))
            cond = tf.tile(tf.reshape(cond,[1,h,w,1]),[minibatch_size,1,1,3])
            zeros = tf.zeros_like(x)
            x = tf.where(cond, zeros, x)
    if crop_pixels > 0:
        with tf.name_scope('Crop'):
            n_crop = crop_pixels
            image_size = x.shape.as_list()[2]
            dmin = tf.random.uniform([2], minval=0, maxval=n_crop, dtype=tf.int32)
            dmax = tf.random.uniform([2], minval=image_size-n_crop, maxval=image_size+1, dtype=tf.int32)
            x = x[:,dmin[0]:dmax[0],dmin[1]:dmax[1],:]
    # with tf.name_scope('Contrast'):
    #     x = tf.image.random_contrast(x, lower=0.5, upper=1.0)
    # with tf.name_scope('Saturation'):
    #     x = tf.image.random_saturation(x, lower=0.5, upper=1.0)
    # with tf.name_scope('Brightness'):
    #     x = tf.image.random_brightness(x, max_delta=0.1)
    # with tf.name_scope('Hue'):
    #     x = tf.image.random_hue(x, max_delta=0.1)
    return x

#----------------------------------------------------------------------------
