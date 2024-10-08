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
import tensorflow as tf
import autokeras as ak

spec_dir = 'data/specs/'


train_ds, test_ds = keras.utils.image_dataset_from_directory(
    spec_dir,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=256,
    image_size=(256, 256),
    validation_split=0.2,
    seed=123,
    subset="both",
    )

#tried autokeras package for image regression - didn't get good results, maybe do more research?

reg = ak.ImageRegressor(overwrite=True)
reg.fit(train_ds, epochs=2)
predicted_y = reg.predict(test_ds, verbose=1)
print(predicted_y)
print(reg.evaluate(test_ds))


