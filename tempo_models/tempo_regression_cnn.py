import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
from keras import models
import librosa
from librosa import display
import tensorflow_datasets as tfds
import tensorflow_io as tfio
import shutil
from get_data import track_tempo, fill_rounded_dict, rounded_dict, get_tempo_spec

spec_dir = 'data/specs/'

fill_rounded_dict()

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    spec_dir,
    labels='inferred',
    label_mode='int', #data only belong to one label
    color_mode='rgb',
    batch_size=64, #edited batch size
    image_size=(256, 256),
    seed=123,
    validation_split=0.2, #20% to validation
    subset="both" #return both training and validation datasets
)

class_names = train_ds.class_names

#break validation into testing and validation datasets
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

normalization_layer = layers.Rescaling(scale=1./255) #standardize rgb values to be 0-1

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

num_classes = len(class_names)

model = Sequential(
  layers.Rescaling(scale=1./255, input_shape=(256,256,3))
)

#adding layers
#tried other way to add layers with model.add()
#edited layers and nodes as I saw results
model.add(keras.Input(shape=(256, 256, 4)))
model.add(layers.Conv2D(32, 5, activation="relu"))
model.add(layers.Conv2D(64, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(64, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Conv2D(128, 3, activation="relu"))
model.add(layers.Conv2D(256, 3, activation="relu"))
model.add(layers.MaxPooling2D(2))
model.add(layers.GlobalMaxPooling2D())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(1, activation="linear"))



model.compile(optimizer='adam',
              loss="MSE")



model.summary()

epochs=20 #edited how long trained for

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)



model.evaluate(test_ds, return_dict=True)