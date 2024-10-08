import tensorflow as tf
import numpy as np
from pydub import AudioSegment
import os
from get_data import echonest
from get_data import get_tempo
from dotenv import load_dotenv
from get_data import get_echonest_track_id
from shutil import move

load_dotenv()


AUDIO_DIR = os.environ.get('AUDIO_DIR')

def convert_to_wav(mp3):
   dst = mp3[0:-3] + "wav"
   sound = AudioSegment.from_mp3(mp3)
   sound.export(dst, format="wav")
   move(dst, 'data/wav')
  
def write_tracks():
    with open('tracks.txt', 'w') as f:
       for track in echonest['track_id']:
              track=str(track)
              f.write(track)
              f.write('\n')
    f.close()
              
def write_tempos():
    tracks = open('tracks.txt', 'r')
    tempos = open('tempos.txt', 'w')
    for track in tracks:
        tempo = str(get_tempo(track))
        tempos.write(tempo)
        tempos.write('\n')
    tracks.close()
    tempos.close()
    

def move_files():
    for i in range(100,155):
        for file in os.listdir(AUDIO_DIR+'/'+str(i)):
            if(file == '.DS_Store'):
                continue
            tracks = open('tracks.txt', 'r')
            for track in tracks:
                if(int(file[0:-4]) == int(track)):
                    move(AUDIO_DIR+'/'+str(i)+'/'+file, 'data/filtered_mp3')


def is_missing(mp3):
    for wav in os.listdir('data/wav'):
        if(int(mp3[0:-4]) == int(wav[0:-4])):
            return False
    return True

def missing_file(track):
    for wav in os.listdir('data/wav'):
        if(int(wav[0:-4])==track):
            return False
        return True

def wav():
    count = 0
    for mp3 in os.listdir('data/filtered_mp3'):
        missing = is_missing(mp3)
        if(missing == True):
            convert_to_wav('data/filtered_mp3/'+ mp3)
