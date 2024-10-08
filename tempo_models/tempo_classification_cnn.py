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
    labels='inferred', #label names coming from folder names
    label_mode='int', #data only belong to one label
    color_mode='rgb',
    batch_size=64, 
    image_size=(55, 166), #tried new image size
    seed=123,
    validation_split=0.2,
    subset="both"
)

class_names = train_ds.class_names

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

normalization_layer = layers.Rescaling(scale=1./255) #standardize rgb values to be 0-1

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
#image_batch, labels_batch = next(iter(normalized_ds))

num_classes = len(class_names)

#edited number of nodes and layers
model = Sequential([
  layers.Rescaling(scale=1./255, input_shape=(55, 166, 3)),
  layers.Conv2D(8, 3, activation='relu'),
  layers.Conv2D(16, 3, activation='relu'),
  layers.MaxPooling2D(2),
  layers.Conv2D(32, 3, activation='relu'),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(2),
  layers.Conv2D(128, 3, activation='relu'),
  layers.Conv2D(256, 3, activation='relu'),
  layers.Flatten(),
  layers.Dense(units=512, activation='relu'),
  layers.Dense(units=256, activation='relu'),
  layers.Dense(units=64, activation='relu'),
  layers.Dense(units=num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#model.build()

model.summary()

epochs=20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

#show graphs
#plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

model.evaluate(test_ds, return_dict=True)