import librosa
import numpy as np
import matplotlib.pyplot as plt

audio_file_path = "data\song2\Mori_Calliope_-_NEZUMI_Scheme.egg"
y, sr = librosa.load(audio_file_path)
n_fft = 2048
hop_length = 512
n_mels = 128

mel_spectrogram = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

spectrogram_data = mel_spectrogram_db
waveform_data = y

fig, ax = plt.subplots(figsize=(15, 7.5))
librosa.display.specshow(
    mel_spectrogram_db, y_axis='mel', x_axis='time', sr=sr)
plt.savefig(f"{audio_file_path[:-4]}melspec.png", dpi=300,
            bbox_inches='tight', pad_inches=0)


spectral_flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
# spectral_flux = librosa.onset.onset_strength(S=spectrogram_data, sr=sr, hop_length=hop_length)

frame_time = librosa.frames_to_time(np.arange(len(spectral_flux)),
                                    sr=sr,
                                    hop_length=hop_length)

fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(14, 6))

ax.plot(frame_time, spectral_flux, label='Spectral flux')
ax.set_title('Spectral flux', fontsize=15)
plt.savefig(f"{audio_file_path[:-4]}spectral_flux.png", dpi=300,
            bbox_inches='tight', pad_inches=0)

tempogram = librosa.feature.tempogram(
    onset_envelope=spectral_flux, sr=sr, hop_length=hop_length)

tempo = librosa.feature.rhythm.tempo(onset_envelope=spectral_flux, sr=sr,
                                     hop_length=hop_length)[0]
fig, ax = plt.subplots(nrows=1, figsize=(15, 10))
librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='tempo', cmap='magma',
                         ax=ax)
ax.axhline(tempo, color='w', linestyle='--', alpha=1,
           label='Estimated tempo={:g}'.format(tempo))
ax.legend(loc='upper right')
ax.set_title('Tempogram', fontsize=15)
plt.savefig(f"{audio_file_path[:-4]}tempogram.png", dpi=300,
            bbox_inches='tight', pad_inches=0)


fig, ax = plt.subplots(figsize=(15, 5))
librosa.display.waveshow(y, sr=sr)
plt.savefig(f"{audio_file_path[:-4]}waveform.png", dpi=300,
            bbox_inches='tight', pad_inches=0)

np.save(f"{audio_file_path[:-4]}melspec.npy", spectrogram_data)
np.save(f"{audio_file_path[:-4]}waveform.npy", waveform_data)
