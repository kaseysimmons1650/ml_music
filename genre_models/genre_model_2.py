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


spec_dir = 'data/specs_genre/'

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    spec_dir,
    labels='inferred', #get labels from folder names
    label_mode='categorical',  #classification
    color_mode='rgb',
    batch_size=64, #edited batch size
    image_size=(256, 256),
    seed=123, #seed for shuffling
    validation_split=0.2, #20% to validation dataset
    subset="both" #return training and validation dataset
)

class_names = train_ds.class_names

#break validation dataset into testing and validation datasets
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

normalization_layer = layers.Rescaling(scale=1./255) #standardize rgb values to be 0-1

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

num_classes = len(class_names)

#tried changing layers and nodes to improve accurracy
#tried using less nodes and less max pooling layers with more dense layers
#edited nodes and layers as I saw results
model = Sequential([
  layers.Rescaling(1./255, input_shape=(256, 256, 4)),
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
  layers.Dense(num_classes) #classification - one node for each label
])



model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), #classification
              metrics=['accuracy'])

model.summary()

epochs=20 #edited in fine tuning

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

#show graphs

# plt.figure(figsize=(8, 8))
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


# model.save('genre_model_2')