import pandas as pd
import numpy as np
import os
from training.loader import gen_beats_array
import librosa
from tqdm import tqdm
from datetime import datetime

CSV_PATH = "dataset/beatmaps/color_notes/combined_songs.csv"  # Path to your CSV file
NPZ_DIR = "dataset/beatmaps/color_notes/npz"  # Folder containing .npz files
LOG_PATH = "error_log.log"


def get_bpm_info(meta, song):
    bpm = meta["bpm"]

    energy = np.sum(song, axis=0)  # type: ignore
    start_index = np.argmax(energy > 0)

    start_time = librosa.frames_to_time(start_index, sr=22050)

    start = float(start_time * 1000)
    beats = int(4)
    return [(bpm, start, beats)]


def log_error(tag, path, message=""):
    with open(LOG_PATH, "a") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {tag} {path} {message}\n")


value = int(input("Enter update option (default 0): ") or 0)

df = pd.read_csv(CSV_PATH)

if value == 0:
    start_index = int(input("Enter starting index (default 0): ") or 0)

    for index, row in tqdm(
        df.iloc[start_index:].iterrows(), total=len(df) - start_index
    ):
        song = row["song"]
        path = os.path.join(NPZ_DIR, song)

        if not os.path.exists(path):
            print(f"Missing: {path}")
            df.at[index, "frames"] = 0
            log_error("MISSING", path)
            continue

        try:
            data = np.load(path, allow_pickle=True)
            data_dict = dict(data)

            if "song" in data and len(data["song"].shape) == 2:
                frames = data["song"].shape[1]
                df.at[index, "frames"] = frames

                bpm_info = get_bpm_info(row, data["song"])
                data_dict["beats_array"] = gen_beats_array(
                    frames,
                    bpm_info,
                    row["duration"] * 1000,
                )
                np.savez(path, **data_dict)
            else:
                df.at[index, "frames"] = 0
                log_error("INVALID SHAPE", path)

        except Exception as e:
            print(f"Error processing {path}: {e}")
            df.at[index, "frames"] = 0
            log_error("ERROR", path, f"- {e}")

        if index % 100 == 0:  # type: ignore
            df.to_csv(CSV_PATH, index=False)

    df.to_csv(CSV_PATH, index=False)

elif value == 1:
    new_column = "npz_size_mb"
    if new_column not in df.columns:
        df[new_column] = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        song = row["song"]
        path = os.path.join(NPZ_DIR, song)

        # Skip rows and leave 'npz_size_mb' as 0
        if (
            not os.path.exists(path)
            or ("automapper" in row and row["automapper"])
            or ("missing_song" in row and row["missing_song"])
        ):
            continue

        try:
            file_size = os.path.getsize(path)  # in bytes
            size_mb = file_size / (1024 * 1024)
            df.at[index, "npz_size_mb"] = round(size_mb, 3)
        except Exception as e:
            log_error("ERROR", path, f"- {e}")
            # leave as 0

        if index % 100 == 0:  # type: ignore
            df.to_csv(CSV_PATH, index=False)

    df.to_csv(CSV_PATH, index=False)
