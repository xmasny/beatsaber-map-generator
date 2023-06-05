import librosa
import numpy as np
import matplotlib.pyplot as plt

audio_file_path = "data/song0/song.egg"
y, sr = librosa.load(audio_file_path)
n_fft = 2048
hop_length = 512
n_mels = 128

mel_spectrogram = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

spectrogram_data = mel_spectrogram_db
# waveform_data = y

fig, ax = plt.subplots(figsize=(15, 7.5))
librosa.display.specshow(
    mel_spectrogram_db, y_axis='mel', x_axis='time', sr=sr)
# plt.savefig(f"{audio_file_path[:-4]}melspec.png", dpi=300,
#             bbox_inches='tight', pad_inches=0)

# fig, ax = plt.subplots(figsize=(15, 5))
# librosa.display.waveshow(y, sr=sr)
# plt.savefig(f"{audio_file_path[:-4]}waveform.png", dpi=300,
#             bbox_inches='tight', pad_inches=0)

# np.save(f"{audio_file_path[:-4]}melspec.npy", spectrogram_data)
# np.save(f"{audio_file_path[:-4]}waveform.npy", waveform_data)

plt.show()
