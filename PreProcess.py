import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns # for data visualization

# import IPython
# import IPython.display as ipd #To play sound in notebook
# import scipy as sci
# import wave
from pathlib import Path

from scipy.fftpack import fft #Fast Fourier Transformation
from scipy.io import wavfile

import librosa
from librosa import display
import os
import glob

input_path = '../AudioTaggerData/'
train_files_path = input_path + 'FSDKaggle2018.audio_train'
test_files_path = input_path + 'FSDKaggle2018.audio_test'
train_csv_path = (input_path +
                  'FSDKaggle2018.meta/train_post_competition.csv')
test_csv_path = (input_path +
                 'FSDKaggle2018.meta/' +
                 'test_post_competition_scoring_clips.csv')

#scipy.wavfile.read returns rate of wave, and # of data read
filename = '/001ca53d.wav'
# sample_rate, samples = wavfile.read(str(audio_train_file) + filename)
# print(samples)
# print(train.shape)

# Data preprocessing part

df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path)

unique_labels = df_train.label.unique()
num_class = len(unique_labels)
label2index = {label: index for index, label in enumerate(unique_labels)}

print('Label to index:')
print(label2index)

train_dict = pd.Series.from_csv(train_csv_path, header = 0).to_dict()
# train_df = pd.read_csv(train_csv_path, header = 0)
# train_dict = train_df.transpose().to_dict()
print('train dict:')
print(train_dict)

#Indices of manually verified training data
# verified_train = np.array(df_train[df_train.manually_verified == 1].index)
# print(len(verified_train))
# print(len(df_train))

#array of labels in number form (0 = hi-hat, 1 = saxophone, etc)
label_emb_indices = np.array([label2index[label] for label in df_train.label])
# print(plain_y_train)

print('Label emb indices:')
print(label_emb_indices)

# # Approach X uses longer sound, then it uses suppressed
# # confX['sampling_rate'] = 26000
# # sampling_rate = 44100 # Original file sr
# sampling_rate = 32000
# # duration = 4
# duration = 5
# # confX['hop_length'] = 520  # 20ms
# hop_length = 192
# fmin = 20
# fmax = sampling_rate // 2
# # confX['n_mels'] = 48
# n_mels = 128
# # confX['n_fft'] = confX['n_mels'] * 20
# n_fft = 1024
# audio_split = 'dont_crop'
# samples = sampling_rate * duration
# dims = (n_mels, 1 + int(np.floor(samples / hop_length)), 1)


def pre_process(pathname):
    sampling_rate = 32000
    # duration = 4
    # duration = 5
    # confX['hop_length'] = 520  # 20ms
    hop_length = 192
    # fmin = 20
    # fmax = sampling_rate // 2
    fmax = None
    # confX['n_mels'] = 48
    n_mels = 128
    # confX['n_fft'] = confX['n_mels'] * 20
    n_fft = 1024
    # audio_split = 'dont_crop'
    # samples = sampling_rate * duration
    # dims = (n_mels, 1 + int(np.floor(samples / hop_length)), 1)

    # y, sr = librosa.load(pathname, sr = sampling_rate)
    y, sr = librosa.load(pathname, sr = None)
    # y, (trim_begin, trim_end) = librosa.effects.trim(y)


    # Amplitudes of STFT
    stft = np.abs(librosa.stft(y, n_fft = n_fft, hop_length = hop_length,
                               window = 'hann', center = True,
                               pad_mode = 'reflect'))

    print('stft shape:', stft.shape)

    freqs = librosa.core.fft_frequencies(sr = sampling_rate, n_fft = n_fft)
    stft = librosa.perceptual_weighting(stft*2, freqs, ref = 1.0, amin = 1e-10,
                                        top_db = 99.0)

    print('stft shape:', stft.shape)

    # Apply mel filterbank
    # Power param is set to 2 (power) by default
    mel_spect = librosa.feature.melspectrogram(S = stft, sr = sampling_rate,
                                               n_mels = n_mels, fmax = fmax)

    print('mel shape:', mel_spect.shape)

    log_mel_spect = librosa.core.power_to_db(mel_spect)

    print('log mel shape:', log_mel_spect.shape)

    # spectrogram = librosa.feature.melspectrogram(S = stft)
    # Keep spectrogram
    # return np.asarray(spectrogram)
    return np.asarray(log_mel_spect)


# pre_process(audio_train_file + filename)
def get_data(pathname, training = True):
    file_list = glob.glob(os.path.join(pathname, '*.wav'))

    if training:
        data_f = open('Audio.train', 'a')
    else:
        data_f = open('Audio.test', 'a')

    # print(file_list)
    spectrograms, times = [], []
    for i, file in enumerate(file_list):
        print("%04d / %d | %s" % (i + 1, len(file_list), file))

        try:
            spectrogram = pre_process(file)
        except Exception:
            print('Weird, couldnt convert to spectrogram, skipping file')
            continue

        # times.append(spectrogram.shape[1])

        time_restriction = 500
        if time_restriction >= spectrogram.shape[1]:
            pad_amount = time_restriction - spectrogram.shape[1]
            # Use avg or max time
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_amount)),
                                 'minimum')
        else:
            spectrogram = spectrogram[:, :time_restriction]

        spectrogram = spectrogram.transpose()

        print("Spectrogram Shape:", spectrogram.shape)

        # spectrograms.append(spectrogram.astype(np.float32))

        # data_f.write(np.array2string(spectrogram) + '\n\n')
        # np.savetxt(data_f, spectrogram)
        # data_f.write('\n\n\n\n\n')


        # if i > 500:
        #     break
        # if 32 < i < 50:  # 34 is a weird one
        if i % 12 == 0:

            plt.figure("General-Purpose ")
            plt.clf()
            plt.subplots_adjust(right = 0.98, left = 0.1, bottom = 0.1,
                                top = 0.99)
            plt.imshow(spectrogram, origin = "lower",
                       interpolation = "nearest", cmap = "viridis")
            plt.xlabel("%d bins" % spectrogram.shape[1])
            plt.ylabel("%d frames" % spectrogram.shape[0])
            plt.colorbar()
            plt.show()

            # display.specshow(spectrogram, y_axis = 'log', x_axis = 'time')
            #
            # plt.title('Mel Spectrogram')
            # plt.colorbar(format = '%+2.0f dB')
            # plt.tight_layout()
            # plt.show()

            print('Spectrogram:', i)
            print(spectrogram)

    # average_time = np.average(times)
    # print('Average timesteps:', average_time)
    # max_time = np.amax(times)
    # print('Max timesteps:', max_time)

    # return spectrograms
    data_f.close()


def get_labels(pathname):
    file_list = glob.glob(os.path.join(pathname, '*.wav'))
    labels = []
    for i, file in enumerate(file_list):
        label = np.zeros((41,))
        categ = train_dict[file]
        hot_index = label2index[categ]
        label[hot_index] = 1
        labels.append(label)

    return np.array(labels)


get_data(test_files_path)


# def read_audio(conf, pathname):
# def read_audio(pathname):
#     #return audio time series and sampling rate
#     y, sr = librosa.load(pathname, sr = sampling_rate)
#     # trim silence
#     if 0 < len(y):
#         y, _ = librosa.effects.trim(y)  # trim, top_db=default(60)
#     # make it unified length to conf.samples
#     if len(y) > samples:  # long enough
#         # if conf['audio_split'] == 'head':
#         y = y[0:samples]
#     else:  # pad blank
#         padding = samples - len(y)    # add padding at both ends
#         offset = padding // 2
#         y = np.pad(y, (offset, samples - len(y) - offset), 'constant')
#     return y, sr


# def audio_to_melspectrogram(conf, audio_timeseries):
#     spectrogram = librosa.feature.melspectrogram(audio_timeseries,
#                                                  sr=conf['sampling_rate'],
#                                                  n_mels=conf['n_mels'],
#                                                  hop_length=conf['hop_length'],
#                                                  n_fft=conf['n_fft'],
#                                                  fmin=conf['fmin'],
#                                                  fmax=conf['fmax'])
#     #convert spectrogram to decibel
#     spectrogram = librosa.power_to_db(spectrogram)
#     spectrogram = spectrogram.astype(np.float32)
#     return spectrogram


# def show_melspectrogram(mels, conf):
#     librosa.display.specshow(mels, x_axis='time', y_axis='mel',
#                              sr=conf['sampling_rate'], hop_length=conf['hop_length'],
#                             fmin=conf['fmin'], fmax=conf['fmax'])
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Log-frequency power spectrogram')
#     plt.show()


# def read_as_melspectrogram(conf, pathname, debug_display=False):
#     x = read_audio(conf, pathname)
#     mels = audio_to_melspectrogram(conf, x)
#     if debug_display:
#         IPython.display.display(IPython.display.Audio(x, rate=conf['sampling_rate']))
#         show_melspectrogram(mels, conf)
#     return mels


#spectograms are ndarray
# mels1 = read_as_melspectrogram(confLH, audio_train_file + '/' +
#                                df_train.fname[0], debug_display=False)
# mels_LH2 = read_as_melspectrogram(confLH, audio_train_file + '/' +
#                                   df_train.fname[1], debug_display=False)
#
# mels2 = read_as_melspectrogram(confX, audio_train_file + '/' +
#                        df_train.fname[0], debug_display=False)

