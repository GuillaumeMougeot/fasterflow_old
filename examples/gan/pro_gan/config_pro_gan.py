#----------------------------------------------------------------------------
# Datasets

# File name
# desc = 'CASIA_WebFace'; data_path = '../data/CASIA-WebFace/'; data_test_path='../data/lfw/'; image_size = 125
desc = 'Cifar10'; data_path = '../../data/cifar10/'; image_size = 32

image_shape = [image_size, image_size, 3]

# Data initializer
# data_initilize='unpickle_all_data_labels'; data_initilize_kwargs = {'resize_images':True, 'channel_mode':'NHWC', 'save_data':True}
# data_initilize = 'load_data_from_tfrecord'; data_initilize_kwargs = {'filename':'{}cifar10_32.tfrecords'.format(data_path)}
data_initilize = 'prepare_cifar10'; data_initilize_kwargs = {'data_path':data_path}
# data_initilize = 'prepare_CASIA'; data_initilize_kwargs = {'data_path':data_path}; data_test_initilize='prepare_lfw'; data_test_initilize_kwargs={'data_path':data_test_path}

# Minibatch selector
data_selector = 'select_minibatch_gan_cifar10'
# data_selector='select_minibatch_rmsg_gan_cifar10'
# data_selector = 'select_minibatch_recognition'

#----------------------------------------------------------------------------
# Training

network = 'pro_gan'; training_function = 'train_pro_gan()'; minibatch_size = {2:128, 3:64, 4:32, 5:32}; stage_length = int(600e3); trans_length = int(600e3)
# network = 'msg_gan'; training_function = 'train_msg_gan()'; minibatch_size = 32
# network = 'arcface'; training_function = 'train_recognition()'; minibatch_size = 64
learning_rate       = 5e-5
optimzer_kwargs     = {'beta1':0.0, 'beta2':0.99}
end_img             = int(12e6)
desc += '-{}'.format(network)

#----------------------------------------------------------------------------
# Config GAN

latent_size = 256

#----------------------------------------------------------------------------
# Saving

nbof_test_sample    = 64
img_per_summary     = int(1e3)
img_per_images      = int(160e3)
img_per_save        = int(800e3)
img_per_val         = int(128e3)
logs_path           = 'logs/' + desc
restore             = None; start_img = 0
# restore             = 'logs/Cifar10-pro_gan-run_20191003192905/models/model-25000.ckpt'; start_img = 25000kimg

#----------------------------------------------------------------------------
