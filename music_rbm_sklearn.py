#This file is heavily based on Daniel Johnson's midi manipulation code in https://github.com/hexahedria/biaxial-rnn-music-composition

import numpy as np
import pandas as pd
import msgpack
import glob
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
from sklearn.neural_network import BernoulliRBM
import sys
###################################################
# In order for this code to work, you need to place this file in the same 
# directory as the midi_manipulation.py file and the Pop_Music_Midi directory

import midi_manipulation

def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e           
    return songs

songs = get_songs(sys.argv[1]) #These songs have already been converted from midi to msgpack
print "{} songs processed".format(len(songs))
###################################################

### HyperParameters

lowest_note = midi_manipulation.lowerBound #the index of the lowest note on the piano roll
highest_note = midi_manipulation.upperBound #the index of the highest note on the piano roll
note_range = highest_note -lowest_note #the note range

num_timesteps  = 60 #This is the number of timesteps that we will create at a time
n_visible      = 2*note_range*num_timesteps #This is the size of the visible layer. 
n_hidden       = 50 #This is the size of the hidden layer

num_epochs = 200 #The number of training epochs that we are going to run. For each epoch we go through the entire data set.
batch_size = 100 #The number of training examples that we are going to send through the RBM at a time. 
lr         = np.float32(0.005) #The learning rate of our model


#### Helper functions. 
model = BernoulliRBM(batch_size=batch_size, learning_rate=lr, n_components=n_hidden, n_iter=num_epochs,
				random_state=None, verbose=0)

training_samples = np.empty([0, 2*note_range*num_timesteps])
for song in songs:
	#The songs are stored in a time x notes format. The size of each song is timesteps_in_song x 2*note_range
	#Here we reshape the songs so that each training example is a vector with num_timesteps x 2*note_range elements
	song = np.array(song)
	song_parts = np.floor(song.shape[0]/num_timesteps)
	song = song[:(int)(song_parts)*num_timesteps]
	song = np.reshape(song, [song.shape[0]/num_timesteps, song.shape[1]*num_timesteps])
	training_samples = np.append(training_samples, song, axis=0)

model.fit(training_samples)
index = np.random.randint(0, training_samples.shape[0])

#This function runs the gibbs chain
def gibbs_sample(k, x_init):
    xk = x_init
    i = 0
    for i in range(0,k):
	xk = model.gibbs(xk)
        print xk.shape
    return xk


#sample = gibbs_sample(10, np.random.choice([0, 1], size=((10, n_visible))))
sample = gibbs_sample(10, np.zeros((10, n_visible)))

final_chords = []
count =0
for i in range(sample.shape[0]):
    if not any(sample[i,:]):
	continue
    #Here we reshape the vector to be time x notes, and then save the vector as a midi file
    S = np.reshape(sample[i,:], (num_timesteps, 2*note_range))
    count+=1
    midi_manipulation.noteStateMatrixToMidi(S, "generated_chord_{}".format(i))
    
    #final_chords.append(np.reshape(sample[i,:], (num_timesteps, 2*note_range)))

#final_chords = np.reshape(final_chords, (num_timesteps, 2*note_range + count))
#midi_manipulation.noteStateMatrixToMidi(final_chords, "generated_chord_final")
