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
    color_mode='rgba',
    batch_size=64, 
    image_size=(55, 166), #tried diff image size
    seed=123,
    validation_split=0.2,
    subset="both"
)

class_names = train_ds.class_names

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

normalization_layer = layers.Rescaling(scale=1./255) #standardize rgb values to be 0-1

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

num_classes = len(class_names)

#edited layers as saw results
model = Sequential([
  layers.Rescaling(1./255, input_shape=(55, 166, 4)),
  layers.Conv2D(3, 3, activation='relu'),
  layers.Conv2D(3, 3, activation='relu'),
  layers.MaxPooling2D(2),
  layers.Conv2D(3, 3, activation='relu'),
  layers.Conv2D(3, 3, activation='relu'),
  layers.MaxPooling2D(2),
  layers.Conv2D(3, 3, activation='relu'),
  layers.Conv2D(3, 3, activation='relu'),
  layers.Flatten(),
  layers.Dense(units=512, activation='relu'),
  layers.Dense(units=256, activation='relu'),
  layers.Dense(units=64, activation='relu'),
  layers.Dense(units=252),
  layers.Dense(units=1, activation="linear")
])

model.compile(optimizer='adam',
              loss="MSE")


model.summary()

epochs=20 #edited how long trained for
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)


model.evaluate(test_ds, return_dict=True)