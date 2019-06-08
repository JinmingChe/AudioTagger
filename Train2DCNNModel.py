from keras import models, layers
from keras_preprocessing.image import ImageDataGenerator
import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
# import IPython.display as ipd  # only for IPython notebooks
import pyaudio
import wave


# Hyperparams / config to test:
# ch 7: (advanced) model ensembling
#   Replace Conv2D w/ DepthwiseConv2D (or maybe SeperableConv2D?)
#   Batch normailization
#   Inception
#   Residual connections
#   DenseNet

# Tune HP (# neurons, layers, epochs, batch_size)
# . Dropout, regularization
# . Data augmentation


class LyrParams:
    # Describes, for a layer, # channels/neurons, dropout rate &
    # regularization choice (Lyr = layer)
    def __init__(self, units, dropout = None, reg = None):
        self.units = units
        self.dropout = dropout
        self.reg = reg


# Lab 3 is a good one to look after
def make_model():
    nn = models.Sequential()
    nn.add(layers.Conv2D())

# 1. Create dummy training data (spectrograms)
dummy_samples = 10
dummy_max_timesteps = 500
dummy_train_data = np.random.random((dummy_samples, dummy_max_timesteps))
dummy_train_labels = np.ndarray(["Applause", "Bark", "Bass_drum",
                                 "Burping_or_eructation", "Bus", "Cello",
                                 "Chime", "Clarinet", "Computer_keyboard",
                                 "Cough"])



