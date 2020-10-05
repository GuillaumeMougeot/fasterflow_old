import os, math, sys

import torch
import numpy as np

from midi_utils import midi2pianoroll, pianoroll2midi, keyed_pianoroll2midi, pianoroll_to_keyed_pianoroll
from plotting import save_image
import GLOBAL_VAR

class MidiDataset(torch.utils.data.Dataset): 
    def __init__(self, n_stages, dataset='full', overwrite=False): 
        super(MidiDataset, self).__init__()

        self.n_stages = n_stages

        dataset_filename = dataset + '_' + str(n_stages)

        if not os.path.exists('dataset/' + dataset_filename + '.data.pt'):        
            print('Building new dataset for the first time.')
            self.samples = self.build_keyed_sample_list(dataset)
        elif overwrite: 
            print('Rebuilding dataset and overwriting .data.pt file.')
            self.samples = self.build_keyed_sample_list(dataset)
        else:
            print('Using existing .data.pt file.')
            if GLOBAL_VAR.DEVICE == 'cpu':
                self.samples = torch.load('dataset/' + dataset_filename + '.data.pt', map_location='cpu')
            else:
                self.samples = torch.load('dataset/' + dataset_filename + '.data.pt')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def build_sample_list(self, dataset='/full/'):
        samples = []

        data_dir = '../../../data/'

        all_songs = []
        for (_, _, filenames) in os.walk(data_dir + dataset):
            all_songs.extend(filenames)
            break

        for song in all_songs:
            pianoroll = midi2pianoroll(data_dir + dataset + '/' + song)
            data_stages = []
            edge_size = 4

            for _ in range(self.n_stages):
                data_stage = get_progressive_data(pianoroll, edge_size)
                data_stages.append(data_stage)
                if not os.path.exists(data_dir + '/' + str(edge_size) + 'x' + str(edge_size) + '/'):
                    os.makedirs(data_dir + '/' + str(edge_size) + 'x' + str(edge_size) + '/')                
                keyed_pianoroll2midi(data_stage, data_dir + '/' + str(edge_size) + 'x' + str(edge_size) + '/' + song  + '.' + str(edge_size) + 'x' + str(edge_size) + '.mid')
                save_image(data_stage, data_dir + '/' + str(edge_size) + 'x' + str(edge_size) + '/' + song  + '.' + str(edge_size) + 'x' + str(edge_size) + '.jpg')
                edge_size = edge_size * 2

            samples.append(data_stages)

        return samples

    def build_keyed_sample_list(self, dataset):
        samples = []

        data_dir = 'dataset/'

        all_songs = []
        for (_, _, filenames) in os.walk(data_dir + dataset):
            all_songs.extend(filenames)
            break

        for song in all_songs:
            try:
                pianorolls = midi2pianoroll(data_dir + dataset + '/' + song)
            except Exception:
                continue

            i = 0
            for pianoroll in pianorolls:
                i += 1
                song_piece = song + '_piece_' + str(i) 
                data_stages = []
                n_cols = 4

                for _ in range(10):
                    data_stage = get_progressive_data_keyed(pianoroll, n_cols)
                    data_stages.append(data_stage)
                    if n_cols > GLOBAL_VAR.NOTE_RANGE_KEYED:
                        path = data_dir + '/' + str(GLOBAL_VAR.NOTE_RANGE_KEYED) + 'x' + str(n_cols) + '/' 
                        full_midi_path = path + song_piece  + '.' + str(GLOBAL_VAR.NOTE_RANGE_KEYED) + 'x' + str(n_cols) + '.mid'
                        save_image(data_stage, path + song_piece  + '.' + str(GLOBAL_VAR.NOTE_RANGE_KEYED) + 'x' + str(n_cols) + '.jpg')
                    else: 
                        path = data_dir + '/' + str(n_cols) + 'x' + str(n_cols) + '/'
                        full_midi_path = path + song_piece  + '.' + str(n_cols) + 'x' + str(n_cols) + '.mid'
                        save_image(data_stage, path + song_piece  + '.' + str(n_cols) + 'x' + str(n_cols) + '.jpg')
                    if not os.path.exists(path):
                        os.makedirs(path)                
                    keyed_pianoroll2midi(data_stage, full_midi_path)
                    n_cols *= 2
                samples.append(data_stages)

                sys.stdout.write('\rProcessed ' + str(len(samples)) + ' samples.')
                sys.stdout.flush()

        print('Built ' + str(len(samples)) + ' entries in dataset.')
        torch.save(samples, 'dataset/' + dataset + '.data.pt')
        print('Saved dataset as .data.pt file.')
        return samples

def get_progressive_data(pianoroll, new_edge_size=None):
    # '''
    # Takes in a pianoroll of shape (n, n), not necessarily sparse but very possible, 
    # and returns a matrix of shape (new_edge_size, new_edge_size). 
    # Every cell in this new matrix will contain the average of the according portion of the input matrix. 
    # new_edge_size must be smaller than the original edge size. 
    # '''

    pianoroll_shape = pianoroll.shape
    if new_edge_size == None or pianoroll_shape[0] < new_edge_size or pianoroll_shape[1] < new_edge_size: 
        return pianoroll

    rows_per_newcell = int(pianoroll_shape[0] / new_edge_size)
    cols_per_newcell = int(pianoroll_shape[1] / new_edge_size)
    progressive_pianoroll = np.zeros((new_edge_size, new_edge_size))

    for row_new in range(new_edge_size):
        for col_new in range(new_edge_size):            
            for row_old in range(rows_per_newcell):
                for col_old in range(cols_per_newcell):

                    idx_row = row_new * rows_per_newcell + row_old
                    idx_col = col_new * cols_per_newcell + col_old

                    if pianoroll[idx_row][idx_col] != 0: 
                        progressive_pianoroll[row_new][col_new] = pianoroll[idx_row][idx_col] #TODO use average here, not first value
                        break
                if progressive_pianoroll[row_new][col_new] != 0: break

    return progressive_pianoroll

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
    if edge_size < GLOBAL_VAR.NOTE_RANGE_KEYED:
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

#----------------------------------------------------------------------------   
# Select minibatch

def select_minibatch_gan(images, crt_img, res, minibatch_size):
    assert minibatch_size < len(images)
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
    maxi = np.max(images[choice_idx,size,:,:,:])
    if maxi==0: maxi=1
    return images[choice_idx,size,:,:,:]*2/maxi - 1

#----------------------------------------------------------------------------   
# Test

def test_build_keyed_sample_list():
    samples = build_keyed_sample_list()
    print(samples[0,0,:5,:5,0])
    batch = select_minibatch_gan(samples, 0, 4, 64)
    print(batch.shape)
    print(batch[0,:5,:5,0])

if __name__=='__main__':
    test_build_keyed_sample_list()

#----------------------------------------------------------------------------   
