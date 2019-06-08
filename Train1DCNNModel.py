from keras import models, layers
from keras_preprocessing.image import ImageDataGenerator
import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
# import IPython.display as ipd  # only for IPython notebooks
import pyaudio
import wave

# Lab 3 is a good one to look after
def make_model():
    nn = models.Sequential()

# 1. Create dummy training data (spectrograms)
dummy_samples = 10
dummy_max_timesteps = 500
dummy_train_data = np.random((dummy_samples, dummy_max_timesteps))
dummy_train_labels = np.ndarray(["Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough"])



