import librosa
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import time

start_time = time.time()


tempo_compare = []

for i in range(20):
    song_folder = f"data\song{i}\\"
    info_json = json.load(open(song_folder + "info.dat", "r"))
    original_tempo = info_json["_beatsPerMinute"]
    audio_file_path = song_folder + info_json["_songFilename"]
    y, sr = librosa.load(audio_file_path)
    n_fft = 2048
    hop_length = 512
    n_mels = 229

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    spectrogram_data = mel_spectrogram_db
    waveform_data = y

    fig, ax = plt.subplots(figsize=(15, 7.5))
    librosa.display.specshow(mel_spectrogram_db, y_axis="mel", x_axis="time", sr=sr)
    plt.savefig(
        f"{audio_file_path[:-4]}melspec.png", dpi=300, bbox_inches="tight", pad_inches=0
    )

    spectral_flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # spectral_flux = librosa.onset.onset_strength(S=spectrogram_data, sr=sr, hop_length=hop_length)

    frame_time = librosa.frames_to_time(
        np.arange(len(spectral_flux)), sr=sr, hop_length=hop_length
    )

    fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(14, 6))

    ax.plot(frame_time, spectral_flux, label="Spectral flux")
    ax.set_title("Spectral flux", fontsize=15)
    plt.savefig(
        f"{audio_file_path[:-4]}spectral_flux.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )

    tempogram = librosa.feature.tempogram(
        onset_envelope=spectral_flux, sr=sr, hop_length=hop_length
    )

    tempo = librosa.feature.rhythm.tempo(
        onset_envelope=spectral_flux, sr=sr, hop_length=hop_length
    )[0]
    fig, ax = plt.subplots(nrows=1, figsize=(15, 10))
    librosa.display.specshow(
        tempogram,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="tempo",
        cmap="magma",
        ax=ax,
    )
    ax.axhline(
        tempo,
        color="w",
        linestyle="--",
        alpha=1,
        label="Estimated tempo={:g}".format(tempo),
    )
    ax.legend(loc="upper right")
    ax.set_title("Tempogram", fontsize=15)
    plt.savefig(
        f"{audio_file_path[:-4]}tempogram.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )

    fig, ax = plt.subplots(figsize=(15, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.savefig(
        f"{audio_file_path[:-4]}waveform.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )

    np.save(f"{audio_file_path[:-4]}melspec.npy", spectrogram_data)
    np.save(f"{audio_file_path[:-4]}waveform.npy", waveform_data)

    tempo_difference = original_tempo - tempo
    tempo_difference_percentage = tempo_difference / original_tempo * 100

    tempo_compare.append(
        (
            original_tempo,
            round(tempo, 2),
            round(tempo_difference, 2),
            round(tempo_difference_percentage, 2),
        )
    )
    print(f"Original tempo: {original_tempo}")
    print("Estimated tempo: %0.2f" % tempo)
    print("Tempo difference: %0.2f" % tempo_difference)
    print("Tempo difference percentage: %0.2f" % tempo_difference_percentage)
    print(f"Song {i} done")

pd.DataFrame(
    tempo_compare,
    columns=[
        "Original tempo",
        "Estimated tempo",
        "Tempo difference",
        "Tempo difference percentage",
    ],
).to_csv("tempo_compare.csv", index=False)

end_time = time.time()
runtime = end_time - start_time

print("Runtime:", runtime, "seconds")
