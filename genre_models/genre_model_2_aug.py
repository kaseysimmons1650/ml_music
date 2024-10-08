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
from keras.preprocessing.image import ImageDataGenerator
from get_data import track_tempo, fill_rounded_dict, rounded_dict, get_tempo_spec


spec_dir = 'data/specs_genre/'

#create datasets
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    spec_dir,
    labels='inferred', #get labels from folder names
    label_mode='categorical',  #classification model
    color_mode='rgb',
    batch_size=256, #changed batch size in fine tuning
    image_size=(256, 256),
    seed=123, #seed for shuffling
    validation_split=0.2, #save 20% for validation dataset
    subset="both" #return both validation and training datsets
)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#break validation dataset into testing and validation
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

normalization_layer = layers.Rescaling(scale=1./255) #STANDARDIZE RGB VALUES TO BE 0-1

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

num_classes = len(class_names)

#try to add more data buy augmenting images
#update: didn't improve results
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal",input_shape=(256,256,3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ])

#tried taking away layers and nodes to not overfit to training
#also didn't really improve results
#edited number of nodes in fine tuning
#added dropout layer
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name="outputs")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=25 #edited during fine tuning

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

#show graph
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

# model.save('genre_model_waveform')