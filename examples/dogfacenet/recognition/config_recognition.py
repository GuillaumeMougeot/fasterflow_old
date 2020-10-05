#----------------------------------------------------------------------------
# Datasets

# File name
# desc = 'DogFaceNet'; data_path = '../../data/tfrecords/DogFaceNet_112_train.tfrecords'; nbof_labels=1000; data_test_path='../../data/tfrecords/DogFaceNet_112_test.tfrecords'; image_size = 112
desc = 'DogFaceNet'; data_path = '../../../data/tfrecords/DogFaceNet_112_train.npy'; nbof_labels=1000; data_test_path='../../../data/tfrecords/DogFaceNet_112_test.npy'; image_size = 112
# desc = 'DogFaceNet'; data_path = '../../../data/DogFaceNet/train'; data_test_path='../../../data/DogFaceNet'; image_size = 112

image_shape = [image_size, image_size, 3]

# Minibatch selector
data_selector = 'select_minibatch_recognition'

#----------------------------------------------------------------------------   
# Training

# network = 'recognition'; training_function = "train_triplet_from_images()"
network = 'recognition'; training_function = "train_triplet_from_npy()"
# network = 'recognition_v2'; training_function = "train_from_npy()"
minibatch_size = 50*3 # Has to be a multiple of 3
learning_rate = 1e-4
regularizer_rate=5e-4
emb_size = 512
optimzer_kwargs     = {'beta1':0.9, 'beta2':0.999}
end_img             = int(18e6)
decrease_rate       = int(end_img//6e6)*10
desc += '-{}'.format(network)

#----------------------------------------------------------------------------
# Saving

nbof_test_sample    = 50*3
img_per_summary     = nbof_test_sample
img_per_save        = nbof_test_sample*int(2e3)
img_per_val         = nbof_test_sample*int(1e2)
stage_length        = end_img
logs_path           = 'logs/' + desc
# restore             = None; start_img = 0
restore             = 'logs/DogFaceNet-recognition-run_20191115142948/models/model-2520.ckpt'; start_img = int(2520e3)

#----------------------------------------------------------------------------
