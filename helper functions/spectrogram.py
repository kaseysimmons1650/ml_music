import os
import pathlib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image 
import librosa
from librosa import display
from tensorflow import keras
from keras import layers
from keras import models
from tensorflow import image
from shutil import move
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from get_data import get_tempo_spec, track_tempo, fill_dict

#fill_dict()

wav_dir = 'data/wav'

sample_path = 'data/wav/000002.wav'



# y, sample_rate = librosa.load(sample_path)
# print('y:', y, '\n')
# print('y shape:', np.shape(y), '\n')
# print('Sample rate (KHz):', sample_rate, '\n')
# print(f'Length of audio: {np.shape(y)[0]/sample_rate}')

# plt.figure(figsize=(15, 5))
# librosa.display.waveshow(y=y, sr=sample_rate, color="blue")
# plt.title("Sound wave", fontsize=20)
# plt.show()

# D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
# print('Shape of D object:', np.shape(D))

# Convert amplitude spectrogram to Decibels-scaled spectrogram.

# DB = librosa.amplitude_to_db(D, ref = np.max)

# Creating the spectogram.

# plt.figure(figsize = (16, 6))
# librosa.display.specshow(DB, sr=sample_rate, hop_length=512,
#                          x_axis='time', y_axis='log')
# plt.colorbar()
# plt.title('Decibels-scaled spectrogram', fontsize=20)
# plt.show()


# Convert sound wave to mel spectrogram.

# y, sr = librosa.load(sample_path)

# S = librosa.feature.melspectrogram(y, sr=sr)
# S_DB = librosa.amplitude_to_db(S, ref=np.max)
# plt.figure(figsize=(15, 5))
# librosa.display.specshow(S_DB, sr=sr, hop_length=512,
#                          x_axis='time', y_axis='log')
# plt.colorbar()
# plt.title("Mel spectrogram", fontsize=20)
# plt.savefig('0000002.png')

# def save_spec(path):
#     plt.ioff()
#     y, sr = librosa.load(path)
#     S = librosa.feature.melspectrogram(y, sr=sr)
#     S_DB = librosa.amplitude_to_db(S, ref=np.max)
#     plt.figure(figsize=(15, 5))
#     librosa.display.specshow(S_DB, sr=sr, hop_length=512)
#     path = path[0:-4]
#     #plt.show()
#     plt.savefig(path+'.png', bbox_inches='tight', pad_inches=0)
#     plt.close()
#     move(path+'.png','specs/')
    
def save_labeled_spec(path):
    #plt.ioff()
    y, sr = librosa.load(path)
    S = librosa.feature.melspectrogram(y, sr=sr)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(S_DB, sr=sr, hop_length=512,x_axis='time', y_axis='log')
    path = path[0:-4]
    #plt.show()
    plt.savefig(path+'.png', bbox_inches='tight', pad_inches=0.25)
    #plt.savefig(path+'.png', pad_inches=0)
    # plt.close()
    tempo = get_tempo_spec(path)
    move(path+'.png','data/specs_labeled/'+str(tempo)) 
    
    
def plot_spec(path):
    y, sr = librosa.load(path)
    S = librosa.feature.melspectrogram(y, sr=sr)
    S_DB = librosa.amplitude_to_db(S, ref=np.max) #convert an amplitude spectrogram to decibels
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(S_DB, sr=sr, hop_length=512,
                            x_axis='time', y_axis='log')
    plt.colorbar()
    plt.title("Mel spectrogram", fontsize=20)
    plt.show()
    
    
# def plot_spectrogram(spectrogram, ax):
#     print(spectrogram.shape)
#     #waveform = np.reshape(spectrogram, (2646238))
#     if len(spectrogram.shape) > 2:
#         assert len(spectrogram.shape) == 3
#     spectrogram = np.squeeze(spectrogram, axis=-1)
#     # Convert the frequencies to log scale and transpose, so that the time is
#     # represented on the x-axis (columns)
#     # Add an epsilon to avoid taking a log of zero.
#     log_spec = np.log(spectrogram.T + np.finfo(float).eps)
#     height = log_spec.shape[0]
#     width = log_spec.shape[1]
#     X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
#     Y = range(height)
#     ax.pcolormesh(X, Y, log_spec)


# for wav in os.listdir('data/wav'):
#     save_spec('data/wav/'+wav)

# plt.ioff()

#save_labeled_spec('data/wav/000002.wav')


# path='data/wav'
# for wav in os.listdir('data/wav'):
#     y, sr = librosa.load(path+wav)
#     S = librosa.feature.melspectrogram(y, sr=sr)
#     S_DB = librosa.amplitude_to_db(S, ref=np.max)

# y, sr = librosa.load('data/wav/000002.wav')
# S = librosa.feature.melspectrogram(y, sr=sr)
# S_DB = librosa.amplitude_to_db(S, ref=np.max)
# print(sr)