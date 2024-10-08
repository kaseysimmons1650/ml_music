import tensorflow as tf
import tensorflow_io as tfio
import os
from dotenv import load_dotenv
import pandas as pd
import csv
import librosa
from librosa import display
import matplotlib.pyplot as plt
from shutil import move
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



AUDIO_DIR = os.environ.get('AUDIO_DIR')

METADATA_DIR = os.environ.get('METADATA_DIR')

echonest = pd.read_csv(METADATA_DIR+"/genres.csv", header=0, skiprows=0, usecols=[0, 3],low_memory=False)
#print(echonest)


# ids = echonest["genre_id"]
# titles = echonest["title"]
# i = 0
# dict = {}
# while i in range(0, len(ids)):
#     dict[ids[i]] = titles[i]
#     i+=1

#print(dict)
# print(ids)
# print(titles)

# with open("genre_id.csv", "w") as csvfile:
#     fieldnames = ['genre_id', 'title']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
#     #writer.writeheader()
#     for key, value in dict.items():
#         writer.writerow(key, value)
    

#echonest.to_csv("genre_id.csv", index = False)



# def get_genre_from_id(genre_id):
#     with open('genre_id.csv') as file:
#         reader = csv.reader(file)
#         genre_id = str(genre_id)
#         for row in reader:
#             if row[0] == genre_id:
#                 return row[1]


#print(get_genre_from_id(76))


# echonest = pd.read_csv(METADATA_DIR+"/tracks.csv", header=1, skiprows=1, usecols=[0, 41],low_memory=False)

# # print(echonest)

# echonest.rename({'Unnamed: 41':'genre_id'}, axis='columns', inplace=True)

# # print(echonest)

# echonest.insert(2, "genre_name")

# echonest.to_csv("genres.csv", index = False)


# def get_genre_id_from_track_id(track_id):
#     with open('genres.csv') as file:
#         reader = csv.reader(file)
#         track_id = str(track_id)
#         for row in reader:
#             if row[0] == track_id:
#                 #print(row[1])
#                 #print(len(row[1]))
#                 if(row[1][2] == ","):
#                     return row[1][1]
#                 elif(row[1][2] == "]"):
#                     return row[1][1]
#                 elif(len(row[1]) >= 5 and row[1][3] != ","):
#                     return row[1][1:4]
#                 else:
#                     return row[1][1:3]


# def get_genre_name(track_id):
#     genre_id = get_genre_id_from_track_id(track_id)
#     genre_name = get_genre_from_id(genre_id)
#     return genre_name

def get_genre_from_track(track_id):
    with open('genres.csv') as file:
        reader = csv.reader(file)
        track_id = str(track_id)
        genre = ""
        for row in reader:
            if row[0] == track_id:
                genre = row[2]
    return genre

# genre_id = get_genre_id_from_track_id(144)
# print(genre_id)
# genre = get_genre_from_id(genre_id)
# print(genre)

# list_genres = []

# for i in range(0, 106576):
#     with open('genres.csv') as file:
#         reader = csv.reader(file)
#         track_id = str(i)
#         for row in reader:
#             if row[0] == track_id:
#                 genre_id = get_genre_id_from_track_id(track_id)
#                 genre = get_genre_from_id(genre_id)
#                 if(genre == "None"):
#                     print(str(i) + ": " + genre)
#                 # list_genres.append(genre)
# file.close()




# echonest = pd.read_csv(METADATA_DIR+"/tracks.csv", header=1, skiprows=1, usecols=[0, 41],low_memory=False)

# # print(echonest)

# echonest.rename({'Unnamed: 41':'genre_id'}, axis='columns', inplace=True)

# # print(echonest)

# echonest.insert(2, "genre_name", list_genres)

# # echonest.to_csv("genres.csv", index = False)

# echonest.to_csv("genres.csv", index = False)

# #print(list_genres)

# with open('genres.csv') as file:
#         reader = csv.reader(file)
#         for row in reader:
#             if row[1] == "[]":
#                 print(row)


# list_tracks = []
# with open('tracks.txt') as tracks:
#     for track in tracks:
#         if "\n" in track:
#             list_tracks.append(track[:-1])
#         else:
#             list_tracks.append(track)


#print(type(list_tracks[12]))

# data = pd.read_csv("genres.csv")
# print(data)

# # to_del = []
# for index, row in data.iterrows():
#     #print(index)
#     #print(row.iloc[0])
#     if(str(row.iloc[0]) not in list_tracks):
#          #print(row.iloc[0])
#          print(index)
#          data.drop(index, axis=0, inplace=True)
#          #to_del.append(row.iloc[0])

# print(data)
# data.to_csv("genres_filtered.csv", index=False)
# #print(to_del[0])

# data = pd.read_csv("genres_filtered.csv")

# genre_list = []
# for index, row in data.iterrows():
#     # print(row.iloc[1])
#     # print(type(row.iloc[1]))
#     # print(row.iloc[0])
#     # print(row.iloc[1])
#     # break
#     genre_id = get_genre_id_from_track_id(row.iloc[0])
#     #print(genre_id)
#     genre = get_genre_from_id(genre_id)
#    # print(genre)
#     if(genre == None):
#         print("index " + str(index))
#         print("track id "+ str(row.iloc[0]))
#         print("genre id " + genre_id)
#         #print("genre " +genre)
#         break
#     genre_list.append(genre)

# # print(genre_list)

# data.insert(2, "genre_name", genre_list, allow_duplicates=True)

# data.to_csv("genres_filtered_with_names.csv", index=False)

# genre_id = get_genre_id_from_track_id(17880)
# print(genre_id)
# genre = get_genre_from_id(genre_id)
# print(genre)

# data = pd.read_csv("genres_filtered_with_names.csv")

# data.replace("Old-Time / Historic","Old-Time_Historic", inplace=True)

# data.to_csv("genres.csv", index=False)

# data = pd.read_csv("genres.csv")

# genre_list = []

# for index, row in data.iterrows():
#     genre = row.iloc[2]
#     if genre not in genre_list:
#         genre_list.append(genre)
#         print(genre)
#     # if (genre == "Old-Time / Historic"):
#     #     print(row.iloc[0])
#     # print(row.iloc[2])

# print(genre_list)
# print(len(genre_list))

# for genre in genre_list:
#     path = os.path.join('data/specs_genre/', genre)
#     if(os.path.exists(path) == False):
#         os.mkdir(path)


def save_genre_spec(path):
    plt.ioff()
    y, sr = librosa.load(path)
    S = librosa.feature.melspectrogram(y, sr=sr)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(7, 5))
    librosa.display.specshow(S_DB, sr=sr, hop_length=512,x_axis='time', y_axis='log')
    track_id = path_to_id(path)
    path = path[:-4]
    #plt.show()
    #plt.savefig(path+'.png', bbox_inches='tight', pad_inches=0.25)
    plt.savefig(path+'.png')
    plt.close()
    genre = get_genre_from_track(track_id)
    print(genre)
    move(path+'.png','data/specs_genre/'+str(genre)) 

def path_to_id(path):
    path = path[9:-4]
    #print(path)
    path = int(path)
    return path

    
# print(get_genre_from_track(2))   
    
# for wav in os.listdir('data/wav'):
#     #genre= get_genre_name()
#     save_genre_spec('data/wav/'+wav)
    #break

# for dir in os.listdir('data/specs_genre'):


def count_specs():    
    total = 0
    for dir in os.listdir('data/specs_genre'):    
        if dir == ".DS_Store":
            continue
        for file in os.listdir('data/specs_genre/'+dir):
            # print(file[-4:])
            # break
            if file[-4:] == ".png":
                total+=1
                #print(total)
    print(total)




def check_spec_exists(track):
    track_id = int(track)
    genre = get_genre_from_track(track_id)
    spec = track + ".png"
    path = os.path.join('data/specs_genre/', genre, spec)
    return os.path.exists(path)

# for wav in os.listdir('data/wav'):
#     #print(wav)
#     track = wav[:-4]
#     #print(track)
#     if check_spec_exists(track) == False:
#         print(track)
#         save_genre_spec('data/wav/'+wav)





def save_all_specs_genre():
    for wav in os.listdir('data/wav'):
    #print(wav)
        track = wav[:-4]
    #print(track)
        if check_spec_exists(track) == False:
            print(track)
            save_genre_spec('data/wav/'+wav)
    