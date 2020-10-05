#----------------------------------------------------------------------------
# Datasets

# File name
# desc = 'Cifar10'; data_path = '../../data/tfrecords/cifar10_train.tfrecords'; nbof_labels=10; data_test_path='../../data/tfrecords/cifar10_test.tfrecords'; image_size = 32
desc = 'vggface2'; data_path = '../../data/tfrecords/vggface2_112.tfrecords'; nbof_labels=8631; data_test_path='../../data/tfrecords/lfw_112_bytes.tfrecords'; image_size = 112
# desc = 'vggface2'; data_path = '../../data/tfrecords/Dog-like-CASIA_112_train.tfrecords'; nbof_labels=1000; data_test_path='../../data/tfrecords/lfw_112_bytes.tfrecords'; image_size = 112
# desc = 'vggface2'; data_path = '../../data/tfrecords/vggface2_112.tfrecords'; nbof_labels=8631; data_test_path='../../data/tfrecords/Dog-like-CASIA_112_test.tfrecords'; image_size = 112
# desc = 'DogFaceNet'; data_path = '../../data/tfrecords/DogFaceNet_112_train.tfrecords'; nbof_labels=1000; data_test_path='../../data/tfrecords/DogFaceNet_112_test.tfrecords'; image_size = 112
# desc = 'DogFaceNet'; data_path = '../../data/tfrecords/Dog-like-CASIA_112_train.tfrecords'; nbof_labels=1000; data_test_path='../../data/tfrecords/Dog-like-CASIA_112_test.tfrecords'; image_size = 112
# desc = 'DogFaceNet'; data_path = '../../data/tfrecords/Dog-like-lfw_112_train.tfrecords'; nbof_labels=1000; data_test_path='../../data/tfrecords/Dog-like-lfw_112_test.tfrecords'; image_size = 112

image_shape = [image_size, image_size, 3]

# Data initializer
# data_initilize = 'prepare_cifar10'; data_initilize_kwargs = {'data_path':data_path}
# data_initilize = 'from_tfrecord_read_images_labels'; data_initilize_kwargs = {'data_path':data_path}; data_test_initilize='from_tfrecord_read_images_labels'; data_test_initilize_kwargs={'data_path':data_test_path}
data_initilize = 'from_tfrecord_read_images_labels'; data_initilize_kwargs = {'data_path':data_path, 'read_from_bytes':True}; data_test_initilize='from_tfrecord_read_images_labels'; data_test_initilize_kwargs={'data_path':data_test_path, 'read_from_bytes':True}

# Minibatch selector
data_selector = 'select_minibatch_recognition'

#----------------------------------------------------------------------------   
# Training

# network = 'classification'; training_function = "train_classification(mode='classification')";  minibatch_size = 128; learning_rate = 1e-4; regularizer_rate=5e-4
network = 'recognition'; training_function = "train_classification(mode='recognition')"; minibatch_size = 64; learning_rate = 1e-4; regularizer_rate=5e-4
# network = 'recognition_v2'; training_function = "train_classification_v2(mode='recognition')"; minibatch_size = 64; learning_rate = 1e-4; regularizer_rate=5e-4
emb_size = 512
optimzer_kwargs     = {'beta1':0.9, 'beta2':0.999}
end_img             = int(18e6)
desc += '-{}'.format(network)

#----------------------------------------------------------------------------
# Saving

nbof_test_sample    = 16
img_per_summary     = nbof_test_sample*10
img_per_save        = nbof_test_sample*int(7e3)
img_per_val         = nbof_test_sample*int(1e3)
stage_length        = end_img
decrease_rate       = (end_img // int(6e6)) * 10
logs_path           = 'logs/' + desc
restore             = None; start_img = 0
# restore             = 'logs/vggface2-recognition-run_20191107220015/models/model-final.ckpt'; start_img = int(6000e3)

#----------------------------------------------------------------------------
