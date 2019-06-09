from keras import models, layers
from keras_preprocessing.image import ImageDataGenerator
import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
# import IPython.display as ipd  # only for IPython notebooks
import pyaudio
import wave

# ch 6: (Sequential methods)
# Staley - 1D CNN is preprocessing step before RNN
# . Bidirectional RNNs, recurrent dropout, & stacking RNNS
#
# ch 5: (Convolution)
# Tune HP (# neurons, layers, epochs, batch_size)
# . Dropout, regularization
# . Data augmentation


# Lab 3 is a good one to look after
def make_model(input_shape):
    nn = models.Sequential()
    nn.add(layers.SeparableConv1D(64, 5, activation = 'relu',
                                  input_shape = (None, input_shape[-1])))
    nn.add(layers.BatchNormalization())
    nn.add(layers.SeparableConv1D(64, 5, activation = 'relu'))
    nn.add(layers.BatchNormalization())
    nn.add(layers.MaxPooling1D(3))
    nn.add(layers.Dropout(0.3))

    nn.add(layers.SeparableConv1D(128, 5, activation = 'relu'))
    nn.add(layers.BatchNormalization())
    nn.add(layers.SeparableConv1D(128, 5, activation = 'relu'))
    nn.add(layers.BatchNormalization())
    # nn.add(layers.MaxPooling1D(3))
    # nn.add(layers.Dropout(0.3))
    #
    # nn.add(layers.SeparableConv1D(512, 5, activation = 'relu'))
    # nn.add(layers.BatchNormalization())
    # nn.add(layers.SeparableConv1D(512, 5, activation = 'relu'))
    # nn.add(layers.BatchNormalization())

    nn.add(layers.Bidirectional(layers.LSTM(128, dropout = 0.3,
                                            recurrent_dropout = 0.3,
                                            return_sequences = True)))
    nn.add(layers.Bidirectional(layers.LSTM(128, dropout = 0.3,
                                            recurrent_dropout = 0.3,
                                            return_sequences = True)))
    nn.add(layers.Bidirectional(layers.LSTM(128, dropout = 0.3,
                                            recurrent_dropout = 0.3,
                                            return_sequences = True)))
    nn.add(layers.Bidirectional(layers.LSTM(128, dropout = 0.3,
                                            recurrent_dropout = 0.3)))

    nn.add(layers.Dense(41, activation = 'softmax'))

    return nn


# 1. Create dummy training data (log mel spectrograms)
dummy_samples = 10
dummy_max_timesteps = 128
dummy_num_freq = 500

dummy_train_data = np.random.random((dummy_samples, dummy_max_timesteps,
                                     dummy_num_freq))
# TODO: Make these categorical one-got 41 elems
dummy_train_labels = np.ndarray(["Applause", "Bark", "Bass_drum",
                                 "Burping_or_eructation", "Bus", "Cello",
                                 "Chime", "Clarinet", "Computer_keyboard",
                                 "Cough"])

model = make_model((dummy_max_timesteps, dummy_num_freq))
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


