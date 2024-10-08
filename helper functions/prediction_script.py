
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import librosa
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class_names = ['Blues', 'Chiptune', 'Classical', 'Compilation', 'Electronic', 'Folk', 'Hip-Hop', 'Indie-Rock', 'International', 'Jazz', 'Kid-Friendly', 'Metal', 'Old-Time Historic', 'Pop', 'Post-Punk', 'Post-Rock', 'Psych-Folk', 'Psych-Rock', 'Punk', 'Rock', 'Sound Art', 'Soundtrack', 'Trip-Hop']


def save_unknown_spec(path):
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

#save_unknown_spec(path_audio+"/AIT Prediction.wav")
#save_unknown_spec(path_audio+"/CHFIL Prediction.wav")

# path1 = 'data/predictions/specs/AIT Prediction.png'
# path2 = 'data/predictions/specs/CHFIL Prediction.png'

# img1 = tf.keras.utils.load_img(path1, target_size=(256,256))
# img2 = tf.keras.utils.load_img(path2, target_size=(256,256))

# img_array1 = tf.keras.utils.img_to_array(img1)
# img_array1 = tf.expand_dims(img_array1, 0) # Create a batch

# img_array2 = tf.keras.utils.img_to_array(img2)
# img_array2 = tf.expand_dims(img_array2, 0)


# prediction1 = new_model(img_array1)
# prediction2 = new_model(img_array2)


# y, sr = librosa.load('data/predictions/audio/AIT Prediction.wav', duration=30)
# onset_env = librosa.onset.onset_strength(y, sr=sr)
# tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
# print(tempo)

# y, sr = librosa.load('data/predictions/audio/CHFIL Prediction.wav', duration=30)
# onset_env = librosa.onset.onset_strength(y, sr=sr)
# tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
# print(tempo)


# path1 = 'data/predictions/specs/AIT Prediction.png'
# path2 = 'data/predictions/specs/CHFIL Prediction.png'


# img1 = tf.keras.utils.load_img(path1, target_size=(256,256))
# img2 = tf.keras.utils.load_img(path2, target_size=(256,256))

# img_array1 = tf.keras.utils.img_to_array(img1)
# img_array1 = tf.expand_dims(img_array1, 0) # Create a batch

# img_array2 = tf.keras.utils.img_to_array(img2)
# img_array2 = tf.expand_dims(img_array2, 0)

# prediction1 = genre_model(img_array1)
# prediction2 = genre_model(img_array2)

# print(prediction1)
# print(prediction2)

# score1 = tf.nn.softmax(prediction1[0])
# score2 = tf.nn.softmax(prediction2[0])

#genre = class_names[np.argmax(score1)]
#genre_2 = class_names[np.argmax(score2)
#print(genre)
#print(genre_2)


def tempo_prediction(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(256,256))
    new_model = tf.keras.models.load_model('regression_model')
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    prediction = np.squeeze(new_model(img_array))
    return prediction

def genre_prediction(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(256,256))
    genre_model = tf.keras.models.load_model('genre_model')
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    prediction = genre_model(img_array)
    score = tf.nn.softmax(prediction[0])
    result = class_names[np.argmax(score)]
    return result