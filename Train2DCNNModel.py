from keras import models, layers
from keras_preprocessing.image import ImageDataGenerator
import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
# import IPython.display as ipd  # only for IPython notebooks
import pyaudio
import wave

# #1 preprocess method
# Normalize input w/ batch normalization (BN)
# Input is trnasformed into logmel domain (turned into log-mel spectrograms)
# BN again & reshaped (time, frequency, 1) - greyscale image
# Then conv stuff

# Hyperparams / config to test:
# ch 7: (advanced)
#   model ensembling (of 2D CNN & Combined 1D CNN & RNN combo, etc??)
#   Replace Conv2D w/ SeparableConv2D
#   Batch normailization (can replace regularization)
#   Inception
#   Residual connections
#   DenseNet is a good model to follow (top leader) or fine tune from

# ch 5: (convnets)
#   Tune HP (# neurons, layers, epochs, batch_size)
#   dropout, regularization (can be replaced by BN)
#   Data augmentation

# K-Fold Validation


# class LyrParams:
#     # Describes, for a layer, # channels/neurons, dropout rate &
#     # regularization choice (Lyr = layer)
#     def __init__(self, units, dpt = None):
#         self.units = units
#         self.dpt = dpt

# nn.add(layers.Conv2D(prm_lyrs[1].units, (3, 3), activation = 'relu'))
# if prm_lyrs[0].dpt:
#   nn.add(layers.dpt(prm_lyrs[0].dpt))

# Lab 3 is a good one for backup
# Model structure after best methods - 6 conv units (conv *2 + maxpool)
def make_model(input_shape):
    nn = models.Sequential()
    nn.add(layers.SeparableConv2D(64, (3, 3), activation = 'relu',
                                  input_shape = input_shape))
    nn.add(layers.BatchNormalization())
    nn.add(layers.SeparableConv2D(64, (3, 3), activation = 'relu'))
    nn.add(layers.BatchNormalization())
    nn.add(layers.MaxPooling2D((2, 2)))
    nn.add(layers.Dropout(0.3))

    nn.add(layers.SeparableConv2D(128, (3, 3), activation = 'relu'))
    nn.add(layers.BatchNormalization())
    nn.add(layers.SeparableConv2D(128, (3, 3), activation = 'relu'))
    nn.add(layers.BatchNormalization())
    nn.add(layers.MaxPooling2D((2, 2)))
    nn.add(layers.Dropout(0.3))

    nn.add(layers.SeparableConv2D(256, (3, 3), activation = 'relu'))
    nn.add(layers.BatchNormalization())
    nn.add(layers.Dropout(0.3))
    nn.add(layers.SeparableConv2D(256, (3, 3), activation = 'relu'))
    nn.add(layers.BatchNormalization())
    nn.add(layers.Dropout(0.3))
    nn.add(layers.SeparableConv2D(256, (3, 3), activation = 'relu'))
    nn.add(layers.BatchNormalization())
    nn.add(layers.Dropout(0.3))
    nn.add(layers.SeparableConv2D(256, (3, 3), activation = 'relu'))
    nn.add(layers.BatchNormalization())
    nn.add(layers.MaxPooling2D((2, 2)))
    nn.add(layers.Dropout(0.3))

    nn.add(layers.SeparableConv2D(512, (3, 3), activation = 'relu'))
    nn.add(layers.BatchNormalization())
    nn.add(layers.SeparableConv2D(512, (3, 3), activation = 'relu'))
    nn.add(layers.BatchNormalization())
    nn.add(layers.MaxPooling2D((2, 2)))
    nn.add(layers.Dropout(0.3))

    nn.add(layers.SeparableConv2D(512, (3, 3), activation = 'relu'))
    nn.add(layers.BatchNormalization())
    nn.add(layers.SeparableConv2D(512, (3, 3), activation = 'relu'))
    nn.add(layers.BatchNormalization())
    nn.add(layers.GlobalAveragePooling2D())

    nn.add(layers.Dense(41, activation = 'softmax'))
    return nn


# 1. Create dummy training data (log mel spectrograms)
dummy_samples = 10
dummy_max_timesteps = 128
dummy_num_freq = 500

dummy_train_data = np.random.random((dummy_samples, dummy_max_timesteps,
                                     dummy_num_freq, 1))
# TODO: Make these categorical one-got 41 elems
dummy_train_labels = np.ndarray(["Applause", "Bark", "Bass_drum",
                                 "Burping_or_eructation", "Bus", "Cello",
                                 "Chime", "Clarinet", "Computer_keyboard",
                                 "Cough"])

nn = make_model((dummy_max_timesteps, dummy_num_freq, 1))
nn.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',
           metrics = ['accuracy'])













