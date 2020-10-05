import os, math, sys
import numpy as np
from midi_utils import midi2pianoroll, pianoroll2midi, keyed_pianoroll2midi, pianoroll_to_keyed_pianoroll
from fasterflow import utils

#----------------------------------------------------------------------------   
# Init dataset

def build_keyed_sample_list(dataset_path='/full/'):
    samples = []

    data_dir = '../../../data/'
    print('Preparing dataset...')
    all_songs = []
    for (_, _, filenames) in os.walk(data_dir + dataset_path):
        all_songs.extend(filenames)
        break

    for song in all_songs:
        try:
            pianorolls = midi2pianoroll(data_dir + dataset_path + '/' + song)
        except Exception:
            continue

        
        
        for pianoroll in pianorolls:
            data_stages = []
            n_cols = 4
            last_res = 7
            for _ in range(last_res):
                data_stage = get_progressive_data_keyed(pianoroll, n_cols)
                data_stage = np.array(data_stage)
                if data_stage.shape[0]!=data_stage.shape[1]:
                    zeros = np.zeros((n_cols,n_cols))
                    zeros[:data_stage.shape[0], :data_stage.shape[1]] = data_stage
                    data_stage = zeros
                data_stage = np.expand_dims(data_stage, axis=0)
                data_stage = np.expand_dims(data_stage, axis=-1)
                data_stage = utils.upscale2d_np(data_stage, factor=2**(last_res+1)//n_cols)
                data_stage = np.squeeze(data_stage, axis=0)
                # data_stage = np.tile(data_stage, 3)

                data_stages.append(data_stage)               
                n_cols *= 2
            samples.append(data_stages)
            sys.stdout.write('\rProcessed ' + str(len(samples)) + ' samples.')
            sys.stdout.flush()
        
    samples = np.array(samples)
    print('\nBuilt ' + str(len(samples)) + ' entries in dataset.')
    print(samples.shape)
    return samples

def get_progressive_data_keyed(pianoroll, edge_size=None):
    keyed_pianoroll = pianoroll_to_keyed_pianoroll(pianoroll)
    if edge_size >= keyed_pianoroll.shape[1] and edge_size >= keyed_pianoroll.shape[0]: 
        return keyed_pianoroll

    # downsample rows
    n_rows = edge_size
    n_cols = keyed_pianoroll.shape[1]
    progressive_keyed_pianoroll = np.zeros((n_rows, n_cols))
    rows_per_newcell = int(keyed_pianoroll.shape[0] / n_rows)
    for col in range(n_cols):            
        for row_new in range(n_rows):
            for row_old in range(rows_per_newcell):
                idx_row = row_new * rows_per_newcell + row_old
                if keyed_pianoroll[idx_row][col] != 0:
                    progressive_keyed_pianoroll[row_new][col] = keyed_pianoroll[idx_row][col]
                    break            

    # downsample cols
    if edge_size < 75:
        n_rows = progressive_keyed_pianoroll.shape[0]        
        n_cols = edge_size
        ret = np.zeros((n_rows, n_cols))

        cols_per_newcell = int(progressive_keyed_pianoroll.shape[1] / n_cols)

        for row in range(n_rows):
            for col_new in range(n_cols):            
                for col_old in range(cols_per_newcell):
                    idx_col = col_new * cols_per_newcell + col_old
                    if progressive_keyed_pianoroll[row][idx_col] != 0:
                        ret[row][col_new] = progressive_keyed_pianoroll[row][idx_col]
                        break            
        
        progressive_keyed_pianoroll = ret

    return progressive_keyed_pianoroll

def load_npz(data_path):
    samples = np.load(data_path, allow_pickle=True, encoding='bytes')
    return samples

#----------------------------------------------------------------------------   
# Select minibatch

def select_minibatch_gan(images, crt_img, res, minibatch_size):
    #assert minibatch_size <= len(images)
    # Choose the appropriate indices
    start_idx = crt_img % len(images)
    end_idx = start_idx + minibatch_size
    if end_idx >= len(images):
        choice_idx = np.arange(start_idx,len(images))
        choice_idx = np.append(choice_idx,np.arange(0,end_idx-len(images)))
    else:
        choice_idx = np.arange(start_idx,end_idx)

    size = int(np.log2(res))-2
    # print(np.max(images[choice_idx,size,:,:,:],axis=[1,2,3],keepdims=True))
    maxi = 100
    chosen_images = np.expand_dims([images[i,size] for i in choice_idx], axis=-1)
    img_shape = chosen_images[0].shape
    if img_shape[0]>img_shape[1]:
        chosen_images = np.append(chosen_images,np.zeros((len(choice_idx),img_shape[0],img_shape[0]-img_shape[1],img_shape[2])),axis=2)
    if img_shape[0] < 256:
        chosen_images = utils.upscale2d_np(chosen_images, factor=256//img_shape[0])
    return chosen_images*2/maxi - 1

#----------------------------------------------------------------------------   
# Test

def test_build_keyed_sample_list():
    import matplotlib.pyplot as plt
    samples = build_keyed_sample_list()
    print(samples[0,0].shape)
    print(samples[0,0,:5,:5,0])
    batch = select_minibatch_gan(samples, 0, 4, 1)
    print(batch.shape)
    print(batch[0,:5,:5,0])
    plt.imshow(batch[0,:,:,0])
    plt.show()

if __name__=='__main__':
    test_build_keyed_sample_list()

#----------------------------------------------------------------------------   
