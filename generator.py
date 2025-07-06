import torch
import numpy as np
import matplotlib.pyplot as plt
from dl.models.onsets import OnsetFeatureExtractor
import seaborn as sns
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BERNOULLI_SAMPLING = True  # ← set to False for thresholding (evaluation)


# Load model
model = OnsetFeatureExtractor(
    input_features=n_mels,
    output_features=1,
    dropout=0.3,
    rnn_dropout=0.1,
    enable_condition=True,
    num_layers=2,
    enable_beats=True,
    inference_chunk_length=round(20480 / FRAME),
).to(device)

model.load_state_dict(torch.load("model_epoch_10.pt", map_location=device))
model.eval()

# Load song
song = np.load("dataset/beatmaps/color_notes/npz/song1005_8b36.npz", allow_pickle=True)

# Access all available difficulties
onsets_by_diff = song["onsets"].item()
available_difficulties = list(onsets_by_diff.keys())

num_runs = 3
modes = {
    "threshold": {
        "runs": 1,
        "extractor": lambda pred_binary, probs: np.where(
            pred_binary.squeeze(0).cpu().numpy() > 0
        )[0],
    },
    "bernoulli": {
        "runs": num_runs,
        "extractor": lambda pred_binary, probs: torch.nonzero(
            torch.bernoulli(probs).squeeze(0)
        )
        .squeeze(-1)
        .cpu()
        .numpy(),
    },
}

difficulty_results = {}

for diff_name in DifficultyNumber:
    if not DifficultyNumber[diff_name.name].value:
        continue

    print(f"\n--- Difficulty: {diff_name.name} ---")
    difficulty_number = DifficultyNumber[diff_name.name].value

    data = dict(
        audio=torch.tensor(song["song"]).T.unsqueeze(0),
        beats=torch.tensor(song["beats_array"]).unsqueeze(0),
        condition=torch.tensor([[difficulty_number]]),
    )

    difficulty_results[diff_name] = {}

    for mode_name, config in modes.items():
        extractor = config["extractor"]
        mode_runs = config["runs"]

        run_onset_counts = []
        run_onset_indices = []

        with torch.no_grad():
            for run in range(mode_runs):
                pred_binary, probs = model.predict_with_probs(data)
                onset_indices = extractor(pred_binary, probs)

                run_onset_counts.append(len(onset_indices))
                run_onset_indices.append(onset_indices)

                if mode_runs == 1:
                    print(f"[{mode_name}]: {len(onset_indices)} onsets")
                else:
                    print(f"[{mode_name}] Run {run}: {len(onset_indices)} onsets")

        difficulty_results[diff_name][mode_name] = {
            "counts": run_onset_counts,
            "onsets": run_onset_indices,
        }


# (Optional) Summary
print("\n=== Summary ===")
for diff, res in difficulty_results.items():
    counts = res["bernoulli"]["counts"]
    print(
        f"{diff.name}: min={min(counts)}, max={max(counts)}, avg={np.mean(counts):.2f}"
    )


def jaccard(a, b):
    a_set = set(a.flatten().tolist())
    b_set = set(b.flatten().tolist())
    union = a_set | b_set
    return len(a_set & b_set) / len(union) if union else 0


# Plot Jaccard heatmaps (only for Bernoulli)
fig, axes = plt.subplots(
    len(difficulty_results), 1, figsize=(7, 5 * len(difficulty_results)), squeeze=False
)

for i, (diff, res) in enumerate(difficulty_results.items()):
    mode = "bernoulli"
    onsets = res[mode]["onsets"]
    num_runs = len(onsets)
    matrix = np.zeros((num_runs, num_runs))

    for r1 in range(num_runs):
        for r2 in range(num_runs):
            matrix[r1, r2] = jaccard(onsets[r1], onsets[r2])

    avg_jaccard = (
        np.mean(
            [matrix[r1, r2] for r1 in range(num_runs) for r2 in range(r1 + 1, num_runs)]
        )
        if num_runs > 1
        else 1.0
    )

    ax = axes[i, 0]
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title(f"{diff} - {mode} (avg Jaccard: {avg_jaccard:.2f})")
    ax.set_xlabel("Run")
    ax.set_ylabel("Run")

plt.tight_layout()

# Print Jaccard similarities between threshold and bernoulli
difficulties = []
similarities = []

for diff, res in difficulty_results.items():
    threshold_onsets = set(res["threshold"]["onsets"][0].flatten().tolist())
    bernoulli_onsets_list = res["bernoulli"]["onsets"]

    jaccards = [
        (
            len(threshold_onsets & set(b.flatten().tolist()))
            / len(threshold_onsets | set(b.flatten().tolist()))
            if threshold_onsets | set(b.flatten().tolist())
            else 0
        )
        for b in bernoulli_onsets_list
    ]
    difficulties.append(str(diff))
    similarities.append(np.mean(jaccards))
    print(f"{diff}: Avg Jaccard vs. threshold = {np.mean(jaccards):.2f}")

# Bar plot of similarity scores
plt.figure(figsize=(8, 4))
plt.bar(difficulties, similarities, color="skyblue")
plt.ylim(0, 1)
plt.title("Jaccard Similarity: Bernoulli vs Threshold")
plt.ylabel("Similarity (0–1)")
plt.xlabel("Difficulty")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
