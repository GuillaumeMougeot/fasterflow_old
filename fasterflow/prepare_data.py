import os
import numpy as np
import skimage as sk 
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from fasterflow import utils

#----------------------------------------------------------------------------
# TF recording

class TFRecordWriter:
    def __init__(self, data_path):
        self.writer = tf.io.TFRecordWriter('{}.tfrecords'.format(data_path))
    
    def add_image(self, image, label=None): # Label param is optional
        image = image.astype(np.uint8)
        if label!=None and type(label)!=int: label = int(label)
        feature = {
            'shape':tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
            'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))}
        if label!=None: feature['label']=tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(example.SerializeToString())
    
    def add_bytes_image(self, bytes_image, image_shape, label=None):
        """
        Will store directly the compressed image instead of a numpy array.
        Prefer this function, the storage space is much smaller.
        """
        if label!=None and type(label)!=int: label = int(label)
        feature = {
            'shape':tf.train.Feature(int64_list=tf.train.Int64List(value=image_shape)),
            'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_image]))}
        if label!=None: feature['label']=tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(example.SerializeToString())

#----------------------------------------------------------------------------
# Prepare standard datasets: main_folder/class_folder/class_images.jpg

def prepare_data(data_path, tfrecord_save_path=None, numpy_save_path=None, output_shape=None, use_bytes_format=False):
    filenames   = []
    labels      = []
    crt_label = 0
    print('Loading dataset from {:s} ...'.format(data_path))
    for root,_,files in os.walk(data_path):
        if len(files)>1: # Keep only faces with more than 2 images
            for f in files:
                filenames += [root+'/'+f]
                labels += [crt_label]
            crt_label += 1
    if tfrecord_save_path!=None:
        # Store images and labels in a tfrecord format
        tfrecord_writer = TFRecordWriter(tfrecord_save_path)
        for i in tqdm(range(len(filenames))):
            image = sk.io.imread(filenames[i])
            if len(image.shape)==3:
                if output_shape!= None: image=sk.transform.resize(image, output_shape, preserve_range=True, anti_aliasing=True).astype(np.uint8)
                # if output_shape!=None:
                #     image=sk.transform.resize(image, (250,250,3), preserve_range=True, anti_aliasing=False).astype(np.uint8)
                #     image=image[69:181,69:181,:].astype(np.uint8)
                if len(image.shape)==3 and image.shape==(112,112,3):
                    if use_bytes_format:
                        tmp_dir = '../tmp'
                        if not os.path.isdir(tmp_dir): os.makedirs(tmp_dir)
                        sk.io.imsave(tmp_dir+'/im.jpg',image)
                        with open(tmp_dir+'/im.jpg','rb') as f:
                            tfrecord_writer.add_bytes_image(f.read(),output_shape,labels[i])
                    else:
                        tfrecord_writer.add_image(image,labels[i])
    if numpy_save_path!=None:
        print('Saving numpy in {}'.format(numpy_save_path))
        numpy_save = []
        # Tensorflow method: faster but takes twice more space on the disk
        # im_ph = tf.placeholder(tf.uint8, shape=[None,None,None], name='im_ph')
        # out_string = tf.io.encode_jpeg(im_ph)
        # with tf.Session() as sess:
        for i in tqdm(range(len(filenames))):
            image = sk.io.imread(filenames[i])
            if len(image.shape)==3:
                if output_shape!=None: image=sk.transform.resize(image, output_shape, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                # if output_shape!=None: image=image[69:181,69:181,:].astype(np.uint8)
                if use_bytes_format:
                    # numpy_save += [sess.run(out_string,feed_dict={im_ph:image})]
                    tmp_dir = '../tmp'
                    if not os.path.isdir(tmp_dir): os.makedirs(tmp_dir)
                    sk.io.imsave(tmp_dir+'/im.jpg',image)
                    with open(tmp_dir+'/im.jpg','rb') as f:
                        numpy_save += [f.read()]
                else:
                    numpy_save += [image]
        np.save(numpy_save_path + '.npy', [np.array(numpy_save), np.array(labels, dtype=int)])
    print('Done: dataset loaded.')
    return np.array(filenames), np.array(labels)

#----------------------------------------------------------------------------
# Prepare LFW

def prepare_lfw(data_path, tfrecord_save_path=None, numpy_save_path=None, output_shape=None, use_bytes_format=False):
    print('Loading dataset from {:s} ...'.format(data_path))
    f = open(data_path+'/pairs.txt', 'r')
    path_img = '/lfw-deepfunneled/'
    pairs = []
    labels = []
    for x in f.readlines():
        if x.rfind('\n')>=0:
            x = x[:x.rfind('\n')]
            pair = x.split('\t')
        if len(pair)==3:
            for i in [1,2]:
                while (len(pair[i]) < 4): pair[i] = '0'+pair[i]
            folder = data_path+path_img+pair[0]+'/'+pair[0]+'_'
            file_pair = [folder+pair[1]+'.jpg',folder+pair[2]+'.jpg']
            pairs += file_pair
            labels += [1,1]
        if len(pair)==4:
            for i in [1,3]:
                while (len(pair[i]) < 4): pair[i] = '0'+pair[i]
            folder = data_path+path_img
            file_pair = [folder+pair[0]+'/'+pair[0]+'_'+pair[1]+'.jpg',folder+pair[2]+'/'+pair[2]+'_'+pair[3]+'.jpg']
            pairs += file_pair
            labels += [0,0]
    pairs = np.array(pairs)
    labels = np.array(labels)
    if tfrecord_save_path!=None:
        # Store images and labels in a tfrecord format
        tfrecord_writer = TFRecordWriter(tfrecord_save_path)
        for i in tqdm(range(len(pairs))):
            image = sk.io.imread(pairs[i])
            if len(image.shape)==3:
                if output_shape!=None: image=sk.transform.resize(image, output_shape, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                # if output_shape!=None: image=image[69:181,69:181,:].astype(np.uint8)
                if use_bytes_format:
                    tmp_dir = '../tmp'
                    if not os.path.isdir(tmp_dir): os.makedirs(tmp_dir)
                    sk.io.imsave(tmp_dir+'/im.jpg',image)
                    with open(tmp_dir+'/im.jpg','rb') as f:
                        tfrecord_writer.add_bytes_image(f.read(),output_shape,labels[i])
                else:
                    tfrecord_writer.add_image(image,labels[i])
    if numpy_save_path!=None:
        print('Saving numpy in {}'.format(numpy_save_path))
        numpy_save = []
        # Tensorflow method: faster but takes twice more space on the disk
        # im_ph = tf.placeholder(tf.uint8, shape=[None,None,None], name='im_ph')
        # out_string = tf.io.encode_jpeg(im_ph)
        # with tf.Session() as sess:
        for i in tqdm(range(len(pairs))):
            image = sk.io.imread(pairs[i])
            if len(image.shape)==3:
                if output_shape!=None: image=sk.transform.resize(image, output_shape, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                # if output_shape!=None: image=image[69:181,69:181,:].astype(np.uint8)
                if use_bytes_format:
                    # numpy_save += [sess.run(out_string,feed_dict={im_ph:image})]
                    tmp_dir = '../tmp'
                    if not os.path.isdir(tmp_dir): os.makedirs(tmp_dir)
                    sk.io.imsave(tmp_dir+'/im.jpg',image)
                    with open(tmp_dir+'/im.jpg','rb') as f:
                        numpy_save += [f.read()]
                else:
                    numpy_save += [image]
        np.save(numpy_save_path + '.npy', [np.array(numpy_save), np.array(labels, dtype=int)])
    print('Done: dataset loaded.')
    return pairs, labels

#----------------------------------------------------------------------------
# Prepare Cifar10

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def prepare_cifar10(data_path, add_test=True, tfrecord_save_path=None, data_format='NHWC'):
    data = np.empty((60000,32,32,3)) if data_format=='NHWC' else np.empty((60000,3,32,32))
    labels = np.empty((0))
    print('Loading dataset from {:s} ...'.format(data_path))
    # Add the test set too
    filenames_list=[data_path+'data_batch_{}'.format(i) for i in range(1,6)]+[data_path+'test_batch']
    # Load the data from files
    for i in range(len(filenames_list)):
        batch = unpickle(filenames_list[i])
        images = np.reshape(batch[b'data'], (10000,3,32,32))
        if data_format=='NHWC':
            images = np.transpose(images, (0,2,3,1))
        data[i*10000:(i+1)*10000] = images
        labels = np.append(labels,batch[b'labels'])
    if tfrecord_save_path!=None:
        # Store data and labels in a tfrecord format
        if add_test:
            tfrecord_writer = TFRecordWriter(tfrecord_save_path)
            for i in range(len(data)):
                tfrecord_writer.add_image(data[i],labels[i])
        else:
            tfrecord_writer_train = TFRecordWriter(tfrecord_save_path+'_train')
            for i in range(50000):
                tfrecord_writer_train.add_image(data[i],labels[i])
            tfrecord_writer_test = TFRecordWriter(tfrecord_save_path+'_test')
            for i in range(50000,len(data)):
                tfrecord_writer_test.add_image(data[i],labels[i])
    print('Done: cifar10 dataset prepared.')
    return data, labels

#----------------------------------------------------------------------------
# Prepare DogFaceNet

#############################################################################
# Prepare identification
#############################################################################
def prepare_dogfacenet(data_path, tfrecord_save_path=None, numpy_save_path=None, output_shape=None, use_bytes_format=False):
    print('Loading dataset from {:s} ...'.format(data_path))
    f = open(data_path+'/pairs.txt', 'r')
    path_img = '/test/'
    pairs = []
    labels = []
    for x in f.readlines():
        if x.rfind('\n')>=0:
            x = x[:x.rfind('\n')]
            pair = x.split('\t')
        if len(pair)==3:
            labels += [int(pair[0]),int(pair[0])]
            pairs += [data_path + path_img + pair[1]]
            pairs += [data_path + path_img + pair[2]]
    pairs = np.array(pairs)
    labels = np.array(labels)
    if tfrecord_save_path!=None:
        print('Saving tfrecords in {}'.format(tfrecord_save_path))
        # Store images and labels in a tfrecord format
        tfrecord_writer = TFRecordWriter(tfrecord_save_path)
        for i in tqdm(range(len(pairs))):
            image = sk.io.imread(pairs[i])
            if len(image.shape)==3:
                if output_shape!=None: image=sk.transform.resize(image, output_shape, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                # if output_shape!=None: image=image[69:181,69:181,:].astype(np.uint8)
                if use_bytes_format:
                    tmp_dir = '../tmp'
                    if not os.path.isdir(tmp_dir): os.makedirs(tmp_dir)
                    sk.io.imsave(tmp_dir+'/im.jpg',image)
                    with open(tmp_dir+'/im.jpg','rb') as f:
                        tfrecord_writer.add_bytes_image(f.read(),output_shape,labels[i])
                else:
                    tfrecord_writer.add_image(image,labels[i])
    if numpy_save_path!=None:
        print('Saving numpy in {}'.format(numpy_save_path))
        numpy_save = []
        # Tensorflow method: faster but takes twice more space on the disk
        # im_ph = tf.placeholder(tf.uint8, shape=[None,None,None], name='im_ph')
        # out_string = tf.io.encode_jpeg(im_ph)
        # with tf.Session() as sess:
        for i in tqdm(range(len(pairs))):
            image = sk.io.imread(pairs[i])
            if len(image.shape)==3:
                if output_shape!=None: image=sk.transform.resize(image, output_shape, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                # if output_shape!=None: image=image[69:181,69:181,:].astype(np.uint8)
                if use_bytes_format:
                    # numpy_save += [sess.run(out_string,feed_dict={im_ph:image})]
                    tmp_dir = '../tmp'
                    if not os.path.isdir(tmp_dir): os.makedirs(tmp_dir)
                    sk.io.imsave(tmp_dir+'/im.jpg',image)
                    with open(tmp_dir+'/im.jpg','rb') as f:
                        numpy_save += [f.read()]
                else:
                    numpy_save += [image]
        np.save(numpy_save_path + '.npy', [np.array(numpy_save), np.array(labels, dtype=int)])
    print('Done: dataset loaded.')
    return pairs, labels

#############################################################################
# Prepare detection
#############################################################################
# def prepare_dogface_detect()


#----------------------------------------------------------------------------
# Prepare data

if __name__=='__main__':
    # prepare_cifar10('../data/cifar10/',add_test=False,tfrecord_save_path='../data/tfrecords/cifar10')

    # prepare_data('../data/DogFaceNet/train', '../data/tfrecords/DogFaceNet_112_train', None, (112,112,3), True)
    # prepare_dogfacenet('../data/DogFaceNet', '../data/tfrecords/DogFaceNet_112_test', None, (112,112,3), True)
    # prepare_data('../data/Dog-like-CASIA/train', '../data/tfrecords/Dog-like-CASIA_112_train', None, (112,112,3), True)
    # prepare_dogfacenet('../data/Dog-like-CASIA', '../data/tfrecords/Dog-like-CASIA_112_test', None, (112,112,3), True)
    # prepare_data('../data/Dog-like-lfw/train', '../data/tfrecords/Dog-like-lfw_112_train', None, (112,112,3), True)
    prepare_dogfacenet('../data/Dog-like-lfw', '../data/tfrecords/Dog-like-lfw_112_test', None, (112,112,3), True)

    # prepare_data('../data/DogFaceNet/train', None, '../data/tfrecords/DogFaceNet_112_train', (112,112,3), True)
    # prepare_dogfacenet('../data/DogFaceNet', None, '../data/tfrecords/DogFaceNet_112_test', (112,112,3), True)

    # prepare_data('../data/CASIA-WebFace', '../data/tfrecords/CASIA-WebFace_112_bytes', None, (112,112,3), True)
    # prepare_data('../data/vggface2', '../data/tfrecords/vggface2_cropped_112', None, (112,112,3), True)
    # prepare_lfw('../data/lfw', '../data/tfrecords/lfw_112_bytes', None, (112,112,3), True)

    # filenames, labels = prepare_lfw('../data/lfw')
    # image = sk.io.imread(filenames[0])
    # plt.imshow(image)
    # plt.show()
    # image = sk.transform.resize(image, (112,112,3), preserve_range=True, anti_aliasing=False).astype(np.uint8)
    # plt.imshow(image)
    # plt.show()

#----------------------------------------------------------------------------