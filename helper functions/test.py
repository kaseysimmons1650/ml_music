import tensorflow as tf
import tensorflow_io as tfio
import os
from dotenv import load_dotenv
from functions import convert_to_wav
import pandas as pd
from get_data import get_echonest_track_id


load_dotenv()

AUDIO_DIR = os.environ.get('AUDIO_DIR')

METADATA_DIR = os.environ.get('METADATA_DIR')

# print(AUDIO_DIR)
# audio = tfio.audio.AudioIOTensor(AUDIO_DIR+"000/000002.mp3")
# print(audio)

audio = AUDIO_DIR+"000/000002.mp3"
audioID = "000002"

#echonest = pd.read_csv(METADATA_DIR+"/echonest.csv", header=2, skiprows=1, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],low_memory=False)
#print(echonest.columns)


# echonest.columns = ['track_id', 'acousticness', 'danceability', 'energy',
#        'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence',
#        'album_date', 'album_name', 'artist_latitude', 'artist_location',
#        'artist_longitude', 'artist_name']

#write_tracks()
track = int('000002')
print(get_echonest_track_id(2))


