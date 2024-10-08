import os
import sys
import shutil
import pickle
import zipfile
import subprocess as sp
from datetime import datetime
from tqdm import tqdm, trange
import pandas as pd
import utils
import tensorflow as tf
import tensorflow_io as tfio
import os
from dotenv import load_dotenv
import pandas as pd
import spotipy

load_dotenv()

METADATA_DIR = os.environ.get('METADATA_DIR')

#load csv, cut out data not necessary
# echonest = pd.read_csv(METADATA_DIR+"/echonest.csv", header=2, skiprows=1, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],low_memory=False)
#features = pd.read_csv(METADATA_DIR+"/features.csv", low_memory=False)

#rename headers
# echonest.columns = ['track_id', 'acousticness', 'danceability', 'energy',
#        'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence',
#        'album_date', 'album_name', 'artist_latitude', 'artist_location',
#        'artist_longitude', 'artist_name']

track_tempo= {}


rounded_dict={}

#get trackid from echonest.csv, ret -1 if not in doc
# def get_echonest_track_id(track_num):
#     id = echonest[echonest['track_id']==track_num].index.values
#     if(id.size==0):
#         return -1
#     else:
#         return id

# #get tempo
# def get_tempo(track_num):
#     track_num = int(track_num)
#     index = echonest[echonest['track_id']==track_num].index.values
#     tempo = echonest.loc[index, 'tempo']
#     return tempo.values[0]

def fill_dict():
    track_file = open('tracks.txt')
    tempo_file = open('tempos.txt')
    tracks = track_file.readlines()
    tempos = tempo_file.readlines()
    for i in range(0,13124):
        track_tempo[int(tracks[i])] = float(tempos[i])

def fill_rounded_dict():
    rounded_dict={}
    track_file = open('tracks.txt')
    tempo_file = open('tempos.txt')
    tracks = track_file.readlines()
    tempos = tempo_file.readlines()
    for i in range(0,13123):
        rounded_dict[int(tracks[i])] = round(float(tempos[i]))
    return rounded_dict

def get_tempo_spec(spec):
    tempo = rounded_dict[int(spec)]
    return tempo


#write_tracks()

#write_tempos()

# fill_rounded_dict()

#print(rounded_dict)


# for tempo in rounded_dict.values():
#     path = os.path.join('data/specs_labeled/', str(tempo))
#     if(os.path.exists(path) == False):
#         os.mkdir(path)


# for spec in os.listdir('data/specs'):
#     if(spec[-4:] == ".png"):
#         tempo = get_tempo_spec(spec[0:-4])
#         shutil.move('data/specs/'+spec, 'data/specs/'+str(tempo))
#         print("moved: " + spec + "to: " + str(tempo))