import pandas as pd
import numpy as np
import os
import json
from training.loader import gen_beats_array, get_onset_array
import librosa
from tqdm import tqdm
from datetime import datetime


# for type in ["color_notes", "bomb_notes", "obstacles"]:
for type in ["color_notes"]:

    base_path = f"dataset/beatmaps/{type}"

    CSV_PATH = f"{base_path}/metadata.csv"  # Path to your CSV file
    NPZ_DIR = f"{base_path}/npz"  # Folder containing .npz files
    LOG_PATH = "error_log.log"
    SONGS_PATH = f"dataset/songs/mel229"  # Path to your songs folder

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

    for value in ["beats", "onset", "size", "validate_onsets", "check_data"]:
        df = pd.read_csv(CSV_PATH)
        if value == "beats":

            for index, row in tqdm(
                df.iterrows(), total=len(df), desc=f"Processing {value}"
            ):
                song = row["song"]
                path = os.path.join(NPZ_DIR, song)

                if row["automapper"]:
                    continue

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

        elif value == "size":
            new_column = "npz_size_mb"
            if new_column not in df.columns:
                df[new_column] = 0.0

            for index, row in tqdm(df.iterrows(), total=len(df)):
                song = row["song"]
                path = os.path.join(NPZ_DIR, song)

                # Skip rows and leave 'npz_size_mb' as 0
                if not os.path.exists(path):
                    continue

                try:
                    file_size = os.path.getsize(path)  # in bytes
                    size_mb = file_size / (1024 * 1024)
                    df.at[index, "npz_size_mb"] = round(size_mb, 3)
                except Exception as e:
                    log_error("ERROR", path, f"- {e}")
                    # leave as 0

            df.to_csv(CSV_PATH, index=False)

        elif value == "onset":

            missing_song = df["missing_song"]
            df = df[~missing_song].reset_index(drop=True)
            df = df[~df["automapper"]].reset_index(drop=True)

            for index, row in tqdm(df.iterrows(), total=len(df)):
                song = row["song"]
                path = os.path.join(NPZ_DIR, song)

                if not os.path.exists(path):
                    print(f"Missing: {path}")
                    log_error("MISSING", path)
                    continue

                try:
                    data = np.load(path, allow_pickle=True)
                    data_dict = dict(data)
                    data_dict["bpm"] = row["bpm"]

                    if "song" in data and len(data["song"].shape) == 2:
                        data_dict["onsets"] = get_onset_array(data_dict)
                        np.savez(path, **data_dict)
                    else:
                        df.at[index, "frames"] = 0
                        log_error("INVALID SHAPE", path)

                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    log_error("ERROR", path, f"- {e}")

        elif value == "validate_onsets":
            VALIDATION_LOG = f"{base_path}/onset_validation_issues.csv"
            DIFFICULTIES = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]

            df_filtered = df[~df["automapper"]].reset_index(drop=True)

            issues = []

            for index, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
                song = row["song"]
                path = os.path.join(NPZ_DIR, song + ".npz")
                if not os.path.exists(path):
                    issues.append(
                        {"song": song, "issue": "File missing", "details": ""}
                    )
                    continue

                try:
                    data = np.load(path, allow_pickle=True)
                    data_dict = dict(data)

                    # Check top-level keys
                    missing_keys = [
                        k
                        for k in ["song", "beats_array", "onsets"]
                        if k not in data_dict
                    ]
                    if missing_keys:
                        issues.append(
                            {
                                "song": song,
                                "issue": "Missing keys",
                                "details": json.dumps(missing_keys),
                            }
                        )
                        continue

                    onsets = data_dict["onsets"].item()  # Should be a dict

                    for diff in DIFFICULTIES:
                        expected = bool(row[diff])
                        exists_in_onsets = diff in onsets

                        if expected and not exists_in_onsets:
                            issues.append(
                                {
                                    "song": song,
                                    "issue": f"Missing expected difficulty '{diff}'",
                                    "details": "",
                                }
                            )
                        elif not expected and exists_in_onsets:
                            issues.append(
                                {
                                    "song": song,
                                    "issue": f"Unexpected difficulty '{diff}' present",
                                    "details": "",
                                }
                            )
                        elif exists_in_onsets:
                            missing_fields = []
                            if "onsets_array" not in onsets[diff]:
                                missing_fields.append("onsets_array")
                            if "condition" not in onsets[diff]:
                                missing_fields.append("condition")
                            if missing_fields:
                                issues.append(
                                    {
                                        "song": song,
                                        "issue": f"Missing fields in '{diff}'",
                                        "details": json.dumps(missing_fields),
                                    }
                                )

                except Exception as e:
                    issues.append(
                        {"song": song, "issue": "Exception", "details": str(e)}
                    )

            if issues:
                pd.DataFrame(issues).to_csv(VALIDATION_LOG, index=False)
                print(f"Validation issues logged to: {VALIDATION_LOG}")
            else:
                print("All onset entries validated successfully.")

        elif value == "check_data":
            VALIDATION_LOG = f"{base_path}/onset_validation_issues.csv"
            DIFFICULTIES = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]

            df_filtered = df[~df["automapper"]].copy()

            # Load list of problematic songs from previous validation
            df_issues = pd.read_csv(VALIDATION_LOG)
            df_issues = pd.DataFrame(df_issues["song"].drop_duplicates())

            results = []

            for _, row in tqdm(df_issues.iterrows(), total=len(df_issues)):
                song_id = row["song"]
                info = {"song": song_id}
                meta_row = df_filtered[df_filtered["song"] == song_id]

                if meta_row.empty:
                    info["error"] = "Metadata missing"
                    results.append(info)
                    continue

                meta_row = meta_row.iloc[0]

                npz_path = os.path.join(NPZ_DIR, song_id)
                if not npz_path.endswith(".npz"):
                    npz_path += ".npz"

                if not os.path.exists(npz_path):
                    info["npz_missing"] = True
                    results.append(info)
                    continue

                try:
                    song_data = np.load(npz_path, allow_pickle=True)
                    song_dict = dict(song_data)

                    info["song_in_npz_missing"] = "song" not in song_dict

                    mel_song_path = os.path.join(SONGS_PATH, song_id)
                    if not mel_song_path.endswith(".npz"):
                        mel_song_path += ".npz"
                    info["mel_song_exists"] = os.path.exists(mel_song_path)
                    info["beats_array_exists"] = "beats_array" in song_dict

                    onsets = song_dict.get("onsets", {})
                    if isinstance(onsets, np.ndarray):
                        onsets = onsets.item()

                    for diff in DIFFICULTIES:
                        expected = bool(meta_row[diff])
                        exists_in_onsets = diff in onsets
                        exists_in_root = diff in song_dict

                        onset_data = (
                            onsets.get(diff, {}).get("onsets_array", [])
                            if exists_in_onsets
                            else []
                        )

                        if expected and not exists_in_onsets:
                            onset_npy_path = os.path.join(
                                base_path, diff, f"{song_id}.npy"
                            )
                            info[diff] = (
                                "onset npy exists"
                                if os.path.exists(onset_npy_path)
                                else "onset npy missing"
                            )
                        elif not expected and exists_in_onsets and not exists_in_root:
                            info[diff] = "unexpected"
                        elif expected and exists_in_onsets:
                            if (
                                isinstance(onset_data, np.ndarray)
                                and len(onset_data) == 0
                            ):
                                info[diff] = "empty"
                            else:
                                info[diff] = (
                                    "ok" if exists_in_root else "ok (but not in root)"
                                )
                        elif not expected and exists_in_onsets:
                            if exists_in_root:
                                info[diff] = "unexpected (but in root)"
                            else:
                                info[diff] = "ok (but not in root)"
                        elif not expected and not exists_in_onsets:
                            info[diff] = "ok"

                except Exception as e:
                    info["error"] = str(e)

                results.append(info)
            pd.DataFrame(results).to_csv(
                f"{base_path}/check_data_report.csv", index=False
            )
            print("Check complete. Results saved to 'check_data_report.csv'")
