import math, os, sys

import mido
import numpy as np
from mido import MidiFile, MidiTrack, Message, second2tick
import pypianoroll
from pypianoroll import Multitrack, Track

import GLOBAL_VAR

def midi2pianoroll(file_path, max_beats=GLOBAL_VAR.MAX_BEATS):

    multitrack = pypianoroll.parse(file_path)
    pianoroll = multitrack.get_merged_pianoroll()

    pianorolls = []
    start = 0 
    while len(pianoroll) - start >= max_beats:
        pianoroll_piece = pianoroll[start:(start + max_beats)]    
        pianorolls.append(pianoroll_piece)
        start += max_beats
        break
    return pianorolls
        
def pianoroll2midi(pianoroll, file_path):
    track = Track(pianoroll=pianoroll)
    multitrack = Multitrack(tracks=[track], tempo=GLOBAL_VAR.MIDI_TEMPO) 
    multitrack.to_pretty_midi()
    multitrack.write(file_path)

    # mid = MidiFile()
    # track = MidiTrack()
    # mid.tracks.append(track)
    # prev_note = 60
    # for beat in range(len(pianoroll[0])):
    #     for note_idx in range(len(pianoroll)):
    #         if pianoroll[note_idx][beat] == 1:             
    #             note = note_idx + 41 +7
    #             track.append(Message('note_on', note=note, time=180))
    #             track.append(Message('note_off', note=prev_note, time=1))
    #             prev_note = note

    # mid.save(file_path)

def keyed_pianoroll2midi(keyed_pianoroll, file_path):
    pianoroll = keyed_pianoroll_to_pianoroll(keyed_pianoroll)
    pianoroll2midi(pianoroll, file_path)    

def flat_tensor2keyed_pianoroll(tensor):
    n_cols = min(int(math.sqrt(tensor.shape[0])), GLOBAL_VAR.NOTE_RANGE_KEYED)
    n_rows = int(tensor.shape[0] / n_cols)
    pianoroll = np.zeros((n_rows, n_cols))

    for row in range(n_rows):
        for col in range(n_cols):
            tensor_idx = row * n_cols + col
            if tensor[tensor_idx] < 0: pianoroll[row][col] = 0
            else: pianoroll[row][col] = int(round(tensor[tensor_idx].item()))

    return pianoroll
    
def midi_dir_to_mp3(path):
    if not os.path.exists(path + '/../mp3'):
        os.makedirs(path + '/../mp3')                

    all_midi = []
    for (_, _, filenames) in os.walk(path):
        all_midi.extend(filenames)
        break

    for midi in all_midi:
        # save as .mp3
        midi_path = path + '/' + midi
        mp3_path = path + '/../mp3/' + midi + '.mp3'
        terminal_cmd = "timidity " + midi_path.replace(' ', '\ ') + " -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k " + mp3_path.replace(' ', '\ ')
        os.system(terminal_cmd)

def print_pianoroll(pianoroll):
    print(' ')
    n_beats = pianoroll.shape[0]
    note_range = pianoroll.shape[1] # can't use global value cause this shall work for keyed and non keyed
    for beat in range(n_beats):
        for note in range(note_range):
            sys.stdout.write(' ' + str(int(pianoroll[beat][note])))
        print('')

def keyed_pianoroll_to_pianoroll(keyed_pianoroll, key_minor=GLOBAL_VAR.KEY_MINOR, add_pitch=GLOBAL_VAR.ADD_PITCH):
    n_beats = keyed_pianoroll.shape[0]
    note_range = keyed_pianoroll.shape[1]
    pianoroll = np.zeros((n_beats, GLOBAL_VAR.NOTE_RANGE))

    if key_minor:
        ### C minor
        note_mapping=[    0,   2,   3,   5,   7,   8,  10,
                         12,  14,  15,  17,  19,  20,  22,
                         24,  26,  27,  29,  31,  32,  34,
                         36,  38,  39,  41,  43,  44,  46,
                         48,  50,  51,  53,  55,  56,  58, 
                         60,  62,  63,  65,  67,  68,  70, 
                         72,  74,  75,  77,  79,  80,  82,
                         84,  86,  87,  89,  91,  92,  94,
                         96,  98,  99, 101, 103, 104, 106,
                        108, 110, 111, 113, 115, 116, 118,
                        120, 122, 123, 125, 127]
    else: 
        ### C major
        note_mapping=[    0,   2,   4,   5,   7,   9,  11,
                         12,  14,  16,  17,  19,  21,  23,
                         24,  26,  28,  29,  31,  33,  35,
                         36,  38,  40,  41,  43,  45,  47,
                         48,  50,  52,  53,  55,  57,  59, 
                         60,  62,  64,  65,  67,  69,  71, 
                         72,  74,  76,  77,  79,  81,  83,
                         84,  86,  88,  89,  91,  93,  95,
                         96,  98, 100, 101, 103, 105, 107,
                        108, 110, 112, 113, 115, 117, 119,
                        120, 122, 124, 125, 127]
    # ### C combined 
    # note_mapping= [  0,   2,   3,   4,   5,   7,   8,   9,  10,  11,
    #                 12,  14,  15,  16,  17,  19,  20,  21,  22,  23,
    #                 24,  26,  27,  28,  29,  31,  32,  33,  34,  35,
    #                 36,  38,  39,  40,  41,  43,  44,  45,  46,  47,
    #                 48,  50,  51,  52,  53,  55,  56,  57,  58,  59, 
    #                 60,  62,  63,  64,  65,  67,  68,  69,  70,  71, 
    #                 72,  74,  75,  76,  77,  79,  80,  81,  82,  83,
    #                 84,  86,  87,  88,  89,  91,  92,  93,  94,  95,
    #                 96,  98,  99, 100, 101, 103, 104, 105, 106, 107,
    #                108, 110, 111, 112, 113, 115, 116, 117, 118, 119,
    #                120, 122, 123, 124, 125, 127]

    for beat in range(n_beats):
        for note in range(note_range):
            if keyed_pianoroll[beat][note] != 0:
                mapped_note = min(
                                    int(note_mapping[min(
                                                            note, len(note_mapping) - 1)]
                                                        ) + add_pitch, 
                                    GLOBAL_VAR.NOTE_RANGE - 1
                                )
                pianoroll[beat][mapped_note] = keyed_pianoroll[beat][note]

    return pianoroll 

def pianoroll_to_keyed_pianoroll(pianoroll):
    # pianoroll = np.array(pianoroll)
    n_beats = pianoroll.shape[0]
    keyed_pianoroll = np.zeros((n_beats, GLOBAL_VAR.NOTE_RANGE_KEYED))

    lowest_note_idx = GLOBAL_VAR.NOTE_RANGE
    lowest_note_idx = np.argmin(pianoroll)%GLOBAL_VAR.NOTE_RANGE
    # for beat in range(n_beats):
    #     for note in range(GLOBAL_VAR.NOTE_RANGE):
    #         if pianoroll[beat][note] != 0:
    #             lowest_note_idx = lowest_note_idx if note > lowest_note_idx else note
    
    note_mapping = get_note_mapping_from_lowest_note(lowest_note_idx)
    mapped_notes = [int(note_mapping[note]) for note in  range(GLOBAL_VAR.NOTE_RANGE)]

    keyed_pianoroll[:, mapped_notes] = pianoroll
    # for beat in range(n_beats):
    #     for note in range(GLOBAL_VAR.NOTE_RANGE):
    #         if pianoroll[beat][note] != 0:
    #             mapped_note = int(note_mapping[note])
    #             keyed_pianoroll[beat][mapped_note] = pianoroll[beat][note]

    return keyed_pianoroll

def enlengthen_pianoroll(pianoroll, factor=GLOBAL_VAR.ENLENGTHEN_FACTOR):
    ret = np.zeros((pianoroll.shape[0] * factor, pianoroll.shape[1]))

    for row in range(pianoroll.shape[0]):
        for col in range(pianoroll.shape[1]):
            for i in range(factor):
                ret[row * factor + i][col] = pianoroll[row][col]

    return ret

def remove_pauses(pianoroll):
    ret = []

    for row in pianoroll:
        if sum(row) != 0:
            ret.append(row)
    
    ret = np.array(ret)
    return ret

def get_note_mapping_from_lowest_note(lowest_note_idx):
    note_mapping = np.zeros(GLOBAL_VAR.NOTE_RANGE)

    c = lowest_note_idx
    d = c + 2
    e = d + 2
    f = e + 1
    g = f + 2
    a = g + 2
    h = a + 2
        
    try:
        note_mapping[c] = 0
        note_mapping[c+12] = 7
        note_mapping[c+24] = 14
        note_mapping[c+36] = 21
        note_mapping[c+48] = 28
        note_mapping[d] = 1
        note_mapping[d+12] = 8
        note_mapping[d+24] = 15
        note_mapping[d+36] = 22
        note_mapping[d+48] = 28
        note_mapping[e] = 2
        note_mapping[e+12] = 9
        note_mapping[e+24] = 16
        note_mapping[e+36] = 23
        note_mapping[e+48] = 29
        note_mapping[f] = 3
        note_mapping[f+12] = 10
        note_mapping[f+24] = 17
        note_mapping[f+36] = 24
        note_mapping[f+48] = 30
        note_mapping[g] = 4
        note_mapping[g+12] = 11
        note_mapping[g+24] = 18
        note_mapping[g+48] = 31
        note_mapping[a] = 5
        note_mapping[a+12] = 12
        note_mapping[a+24] = 19
        note_mapping[a+36] = 26
        note_mapping[a+48] = 32
        note_mapping[h] = 6
        note_mapping[h+12] = 13
        note_mapping[h+24] = 20
        note_mapping[h+36] = 27
        note_mapping[h+48] = 33
    except IndexError:
        pass

    return note_mapping

