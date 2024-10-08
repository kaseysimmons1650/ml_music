import time
import tensorflow as tf
import tensorflow_io as tfio
import os
from dotenv import load_dotenv
from get_data import fill_rounded_dict, get_tempo_spec
import pandas as pd
import csv
import librosa
from librosa import display
import matplotlib.pyplot as plt
from shutil import move
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

rounded_dict = fill_rounded_dict()
#print(rounded_dict)

def get_tempo_spec(track_id):
    tempo = rounded_dict[track_id]
    return tempo

def path_to_id(path):
    path = path[9:-4]
    #print(path)
    path = int(path)
    return path

def save_wave(wav):
    y, sr = librosa.load(wav, duration=4)
    librosa.display.waveshow(y, sr=sr, color="blue")
    # plt.show()
    track_id = path_to_id(wav)
    tempo = get_tempo_spec(track_id)
    wav = wav[:-4]
    #plt.show()
    plt.savefig(wav+'.png', bbox_inches='tight', pad_inches=0.25)
    move(wav+'.png','data/waveforms_trim/'+str(tempo))
    #plt.savefig(path+'.png')
    #plt.close()
    #genre = get_genre_from_track(track_id)
    #print(genre)
    # tempo = 
    # move(path+'.png','data/specs_genre/'+str(genre)) 
    

def save_tempogram(wav):
    y, sr = librosa.load(wav, duration=30)
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                      hop_length=hop_length)
    librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='tempo', cmap='magma')
    # plt.show()
    track_id = path_to_id(wav)
    tempo = get_tempo_spec(track_id)
    wav = wav[:-4]
    #plt.show()
    plt.savefig(wav+'.png', bbox_inches='tight', pad_inches=0.25)
    move(wav+'.png','data/tempograms/'+str(tempo))

    
# for wav in os.listdir('data/wav'):
#     path = 'data/wav/'+wav
#     track_id = path_to_id(path)
#     print(track_id)
#     tempo = get_tempo_spec(track_id)
#     print(tempo)
#     plt.figure(figsize=(8, 6))
#     y, sr = librosa.load(path, duration=30)
#     librosa.display.waveshow(y, sr=sr, color="blue")
#     wav = wav[:-4]
#     # plt.show()
#     plt.savefig(wav+'.png', bbox_inches='tight', pad_inches=0.25)
#     move(wav+'.png','data/waveforms/'+str(tempo))
    #break 
   

# for dir in os.listdir('data/specs'):
#     if dir == ".DS_Store":
#         continue
#     path = os.path.join('data/waveforms_trim/',dir)
#     if(os.path.exists(path) == False):
#         os.mkdir(path)

# for dir in os.listdir('data/specs'):
#     if dir == ".DS_Store":
#         continue
#     path = os.path.join('data/tempograms/',dir)
#     if(os.path.exists(path) == False):
#         os.mkdir(path)

def check_wave_exists(track):
    track_id = int(track)
    #print(track_id)
    tempo = get_tempo_spec(track_id)
    wave = track + ".png"
    path = os.path.join('data/waveforms_trim/', str(tempo), wave)
    return os.path.exists(path)

def check_tempogram_exists(track):
    track_id = int(track)
    #print(track_id)
    tempo = get_tempo_spec(track_id)
    #print(tempo)
    wave = track + ".png"
    path = os.path.join('data/tempograms/', str(tempo), wave)
    return os.path.exists(path)

# for wav in os.listdir('data/wav'):
#     #print(wav)
#     track = wav[:-4]
#     #print(track)
#     if check_wave_exists(track) == False:
#         print(track)
#         save_wave('data/wav/'+wav)

# count = 0
# for dir in os.listdir('data/tempograms'):
#     for file in os.listdir('data/tempograms/'+dir):
#         count +=1
#         #print(dir + "/" + file)
# print(count)


for wav in os.listdir('data/wav'):
    #print(wav)
    track = wav[:-4]
    #print(track)
    if check_tempogram_exists(track) == False:
        print(track)
        save_tempogram('data/wav/'+wav)