import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
# import IPython.display as ipd  # only for IPython notebooks
# import pyaudio
# import wave

# 1. Get the file path to the included audio example
# filename = librosa.util.example_audio_file()
filename = '../AudioTaggerData/FSDKaggle2018.audio_test/0a23fde2.wav'

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load(filename, sr=None)
print('Waveform:\n', y, '\nSampling Rate (Hz):', sr)

print('Len of waveform:', len(y))

# 3. Create the short-term fourier transfrom of 'y'
d = librosa.stft(y)
# np.abs(D[f, t]) is the magnitude of frequency bin f at frame t
# visual_d = np.abs(d)
print('\nSpectrogram Data: ( shape =', np.abs(d).shape, ')\n')
print(np.abs(d))
print()
print('(magnitude, timesteps)')

# Display spectrogram
display.specshow(librosa.amplitude_to_db(np.abs(d), ref=np.max),
                 y_axis='log',
                 x_axis='time')

plt.title('Power Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
# plt.show()  # Toggle show plot

# 4. Check: reconstruct waveform from short-term fourier transfrom
#    Turn reconstructed wavefrom into new WAV file to play
rec_y = librosa.istft(d)
# print(len(y), len(rec_y))
reconst_filename = 'recon_0a5cbf90.wav'
librosa.output.write_wav(reconst_filename, rec_y, sr)

# Play a WAV file, comment out below to toggle
# ipd.Audio(y, sr)  # only for IPython notebooks

# p = pyaudio.PyAudio()
# wf = wave.open(filename, 'rb')
# stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#                 channels=wf.getnchannels(),
#                 rate=wf.getframerate(),
#                 output=True)
# data = wf.readframes(1024)
# while data != '':
#     stream.write(data)
#     data = wf.readframes(1024)

# stream.stop_stream()
# stream.close()
# p.terminate()

# Must ^C to end b/c wf.readframes() doesn't stop
