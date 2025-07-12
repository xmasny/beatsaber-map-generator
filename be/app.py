import os
from pathlib import Path
import random
import re
from flask import Flask, jsonify, request
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)

BASE_PATH = Path("dataset/batch")


@app.route("/api")
def hello():
    return "Hello from Flask inside Docker!"


@app.route("/api/claim-seed", methods=["POST"])
def claim_random_seed():
    data = request.get_json()
    difficulty = data.get("difficulty")
    if not difficulty:
        return jsonify({"error": "Missing difficulty"}), 400

    difficulty_path = BASE_PATH / difficulty
    if not difficulty_path.exists() or not difficulty_path.is_dir():
        return jsonify({"error": f"Difficulty path not found: {difficulty_path}"}), 404

    # Get all batch folders under the difficulty that are not marked as used
    unused_batches = [
        p
        for p in difficulty_path.iterdir()
        if p.is_dir() and not (p / ".used").exists()
    ]

    if not unused_batches:
        return jsonify({"error": "No available unused batches"}), 404

    # Randomly select one and mark it as used
    selected = random.choice(unused_batches)
    used_marker = selected / ".used"
    try:
        # used_marker.touch()
        return jsonify({"batch": selected.name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/get-list-of-batches", methods=["POST"])
def get_list_of_batches():
    data = request.get_json()
    difficulty = data.get("difficulty")
    split_seed = data.get("split_seed")
    split = data.get("split", "train")
    model_type = data.get("model_type", "onsets")

    df = pd.read_parquet(
        f"dataset/beatmaps/color_notes/notes_dataset/notes_{difficulty}.parquet"
    )

    df_files = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
    full_train_df, test_df = train_test_split(df_files, test_size=0.3)
    valid_df, test_df = train_test_split(test_df, test_size=0.5)

    sizes = {
        "train": full_train_df.shape[0],
        "validation": valid_df.shape[0],
        "test": test_df.shape[0],
    }

    print(f"Sizes: {sizes}")

    if not difficulty or not split_seed:
        return jsonify({"error": "Missing required fields"}), 400

    path = BASE_PATH / difficulty / split_seed / split / model_type

    print(f"Listing batches in: {path}")

    try:
        full_paths = [os.path.join(str(path), f) for f in os.listdir(path)]
        files_sorted = sorted(
            full_paths,
            key=lambda x: int(re.search(r"batch_(\d+)\.npz", x).group(1)),  # type: ignore
        )
        return jsonify({"files": files_sorted, "no_songs": sizes[split]})
    except FileNotFoundError:
        return jsonify({"error": f"Path not found: {path}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
