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



image_height = 256
image_width = 256
batch_size = 32 #edited batch size
path = 'data/specs/'
num_epochs = 10 #edited how long trained for

# Load dataset using image_dataset_from_directory
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    labels="inferred", #get label names from folder names
    label_mode='int',
    image_size=(image_height, image_width),
    batch_size=batch_size,
    color_mode="rgb",
    validation_split=0.2,
    subset="both",
    seed=42
)

#split validation into validation and testing
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)


#tried less Conv2D and less nodes
#edited as saw results
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation="linear")  # Output layer for regression
])


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


history = model.fit(train_ds, epochs=num_epochs, validation_data=val_ds)


test_loss, test_mae = model.evaluate(test_ds)