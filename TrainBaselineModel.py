from keras import models, layers
from keras_preprocessing.image import ImageDataGenerator
import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
# import IPython.display as ipd  # only for IPython notebooks
import pyaudio
import wave

# Baseline provided from:
# https://github.com/DCASE-REPO/dcase2018_baseline/tree/master/task2

# Lab 3 is a good one to look after
def make_model():
    nn = models.Sequential()
    model.add(layers.Conv2D(100, (7, 7), activation = 'relu', 
                            input_shape = (batch_size, 1025, 71)))
    model.add(layers.MaxPool2D(3, 3), strides = (2, 2))
    model.add(layers.Conv2D(150, (5, 5), activation = 'relu')
    model.add(layers.MaxPool2D(3, 3), strides = (2, 2))
    model.add(layers.Conv2D(200, (3, 3), activation = 'relu')
    # ReduceMax??? keras.backend.max

    

# 1. Create dummy training data (spectrograms)
dummy_samples = 10
dummy_max_timesteps = 500
dummy_train_data = np.random((dummy_samples, dummy_max_timesteps))
dummy_train_labels = np.ndarray(["Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough"])



