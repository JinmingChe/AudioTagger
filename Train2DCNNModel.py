from keras import models, layers
from keras_preprocessing.image import ImageDataGenerator
# import librosa
# from librosa import display
import numpy as np
# import matplotlib.pyplot as plt
# import IPython.display as ipd  # only for IPython notebooks
# import pyaudio
# import wave
import PreProcess

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
    # Current Shape: (500, 128, 1)  # 500 = timesteps, 128 = frequencies
    nn = models.Sequential()
    nn.add(layers.SeparableConv2D(64, (3, 3), padding = 'same',
                                  activation = 'relu',
                                  input_shape = input_shape))
    # Shape: (126, 498, 64)
    nn.add(layers.BatchNormalization())
    nn.add(layers.SeparableConv2D(64, (3, 3), padding = 'same',
                                  activation = 'relu'))
    # Shape: (124, 496, 64)
    nn.add(layers.BatchNormalization())
    nn.add(layers.MaxPooling2D((2, 2)))
    # Shape: (62, 248, 64)
    nn.add(layers.Dropout(0.3))

    nn.add(layers.SeparableConv2D(128, (3, 3), padding = 'same',
                                  activation = 'relu'))
    # Shape: (60, 246, 128)
    nn.add(layers.BatchNormalization())
    nn.add(layers.SeparableConv2D(128, (3, 3), padding = 'same',
                                  activation = 'relu'))
    # Shape: (58, 244, 128)
    nn.add(layers.BatchNormalization())
    nn.add(layers.MaxPooling2D((2, 2)))
    # Shape: (29, 122, 128)
    nn.add(layers.Dropout(0.3))

    # Possibly make this block more like the others
    nn.add(layers.SeparableConv2D(256, (3, 3), padding = 'same',
                                  activation = 'relu'))
    # Shape: (27, 120, 256)
    nn.add(layers.BatchNormalization())
    nn.add(layers.Dropout(0.3))
    nn.add(layers.SeparableConv2D(256, (3, 3), padding = 'same',
                                  activation = 'relu'))
    # Shape: (25, 118, 256)
    nn.add(layers.BatchNormalization())
    nn.add(layers.Dropout(0.3))
    nn.add(layers.SeparableConv2D(256, (3, 3), padding = 'same',
                                  activation = 'relu'))
    # Shape: (23, 116, 256)
    nn.add(layers.BatchNormalization())
    nn.add(layers.Dropout(0.3))
    nn.add(layers.SeparableConv2D(256, (3, 3), padding = 'same',
                                  activation = 'relu'))
    # Shape: (21, 114, 256)
    nn.add(layers.BatchNormalization())
    nn.add(layers.MaxPooling2D((2, 2)))
    # Shape: (10, 57, 256)
    nn.add(layers.Dropout(0.3))

    nn.add(layers.SeparableConv2D(512, (3, 3), padding = 'same',
                                  activation = 'relu'))
    # Shape: (8, 55, 512)
    nn.add(layers.BatchNormalization())
    nn.add(layers.SeparableConv2D(512, (3, 3), padding = 'same',
                                  activation = 'relu'))
    # Shape: (6, 53, 512)
    nn.add(layers.BatchNormalization())
    nn.add(layers.MaxPooling2D((2, 2)))
    # Shape: (3, 26, 512)
    nn.add(layers.Dropout(0.3))

    nn.add(layers.SeparableConv2D(512, (3, 3), padding = 'same',
                                  activation = 'relu'))
    # Shape: (1, 24, 512)
    nn.add(layers.BatchNormalization())
    nn.add(layers.SeparableConv2D(512, (3, 3), padding = 'same',
                                  activation = 'relu'))
    nn.add(layers.BatchNormalization())
    nn.add(layers.GlobalAveragePooling2D())

    nn.add(layers.Dense(41, activation = 'softmax'))
    return nn

# Get data
input_path = '../AudioTaggerData/'
train_files_path = input_path + 'FSDKaggle2018.audio_train'
test_files_path = input_path + 'FSDKaggle2018.audio_test'
train_csv_path = (input_path +
                  'FSDKaggle2018.meta/train_post_competition.csv')
test_csv_path = (input_path +
                 'FSDKaggle2018.meta/' +
                 'test_post_competition_scoring_clips.csv')

# 1. Create training data (log mel spectrograms)
# dummy_samples = 10
max_timesteps = 1000
num_freq = 128

train_data = PreProcess.pre_process(train_files_path)
train_labels = PreProcess.get_labels(train_files_path)

print()

# Use getter methods here
# dummy_train_data = np.random.random((dummy_samples, dummy_max_timesteps,
#                                      dummy_num_freq, 1))
# # Categorical labels
# dummy_train_labels = np.zeros((10, 41))
# for i, label in enumerate(dummy_train_labels):
#     label[i] = 1
# print(dummy_train_labels)

model = make_model((max_timesteps, num_freq, 1))
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

k = 4
num_val = len(train_data) // k
num_train = len(train_labels) - num_val
all_val_acc_histories, all_val_loss_histories = [], []
for x in range(k):
    val_data = train_data[x * num_val: (x + 1) * num_val]
    val_labels = train_labels[x * num_val: (x + 1) * num_val]

    partial_train_data = np.concatenate(
        [train_data[: x * num_val], train_data[(x + 1) * num_val:]],
        axis = 0)
    partial_train_labels = np.concatenate(
        [train_labels[: x * num_val],
         train_labels[(x + 1) * num_val:]],
        axis = 0)

    hst = model.fit(partial_train_data, partial_train_labels, batch_size = 2,
                    epochs = 10, validation_data = (val_data, val_labels),)

    hst = hst.history
    all_val_loss_histories.append(hst['val_loss'])
    all_val_acc_histories.append(hst['val_acc'])

avg_val_loss_hst = np.mean(all_val_loss_histories, axis = 0)
avg_val_acc_hst = np.mean(all_val_acc_histories, axis = 0)

best_loss, best_acc, prev_acc, best_epoch = None, None, None, 0

acc_increased = True
for i in range(10):
    print(avg_val_acc_hst[i], '/', avg_val_loss_hst[i])

    if prev_acc is not None and avg_val_acc_hst[i] < prev_acc:
        acc_increased = False
    prev_acc = avg_val_loss_hst[i]

    if (best_acc is None or avg_val_acc_hst[i] > best_acc and
            acc_increased):
        best_acc = avg_val_acc_hst[i]
        best_loss = avg_val_loss_hst[i]
        best_epoch = i + 1

print('Best val loss:', best_loss, '& with acc:', best_loss, 'at epoch:',
      str(best_epoch))
















