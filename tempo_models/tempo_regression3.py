import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from keras import layers, Sequential
from keras import models
import librosa
import pandas as pd
from librosa import display
import tensorflow_datasets as tfds
import tensorflow_io as tfio
import shutil
from get_data import track_tempo, fill_rounded_dict, rounded_dict, get_tempo_spec

path = 'data/specs/'

#used new way to create datasets!
#best tempo model results!!

image_dir = Path(path)
filepaths = pd.Series(list(image_dir.glob(r'**/*.png')), name="Filepath").astype(str)

tempos = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name="Tempo").astype(np.int64)

images = pd.concat([filepaths, tempos], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)

image_df = images

train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)

#print(images)

#print(train_df)

#create image data generators for each dataset
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

#flow from dataframe - create training dataset
train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Tempo',
    target_size=(256,256),
    color_mode="rgb",
    class_mode="raw",
    batch_size=64,
    shuffle=True,
    seed=42,
    subset="training"
)

#flow from dataframe - create validation dataset
val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Tempo',
    target_size=(256,256),
    color_mode="rgb",
    class_mode="raw", #bc regression
    batch_size=64,
    shuffle=True,
    seed=42,
    subset="validation"
)

#create testing dataset
test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Tempo',
    target_size=(256,256),
    color_mode="rgb",
    class_mode="raw", #bc regression
    batch_size=64,
    shuffle=False,
)


#tried adding more layers and nodes
#edited nodes and layers as saw results
inputs = tf.keras.Input(shape=(256,256,3))
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation="relu")(inputs)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu")(x) 
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu")(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu")(x) 
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation='linear')(x) #bc regression task!

model=tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer="adam",
    loss="mse" #bc regression task
)

history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=25 #edited how long model trained for
)


# predicted_tempos = np.squeeze(model.predict(test_images))
# # squeeze - want one dimensional predictions
# true_tempos = test_images.labels

rmse = np.sqrt(model.evaluate(test_images, verbose=0))
print("Test RMSE: {:.5f}".format(rmse))
# difference in prediction and real value



model.save('regression_model')