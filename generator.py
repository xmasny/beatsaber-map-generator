import torch
import numpy as np
import matplotlib.pyplot as plt
from dl.models.onsets import SimpleOnsets
import seaborn as sns
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SimpleOnsets(
    input_features=n_mels,
    output_features=1,
    dropout=0.4,
    rnn_dropout=0.1,
    enable_condition=True,
    num_layers=2,
    enable_beats=True,
    inference_chunk_length=round(16000 / FRAME),
).to(device)

model.load_state_dict(torch.load("model_model_epoch_30.pt", map_location=device))
model.eval()

# Load song
song = np.load("dataset/beatmaps/color_notes/npz/song1005_8b36.npz", allow_pickle=True)

# Access all available difficulties
onsets_by_diff = song["onsets"].item()
available_difficulties = list(onsets_by_diff.keys())

num_runs = 2
difficulty_results = {}

for diff_name in DifficultyNumber:
    if not DifficultyNumber[diff_name.name].value:
        continue
    print(f"\n--- Difficulty: {diff_name} ---")

    difficulty_number = DifficultyNumber[diff_name.name].value

    data = dict(
        audio=torch.tensor(song["song"]).T.unsqueeze(0),
        beats=torch.tensor(song["beats_array"]).unsqueeze(0),
        condition=torch.tensor([[difficulty_number]]),
    )

    run_onset_counts = []
    run_onset_indices = []

    with torch.no_grad():
        for run in range(num_runs):
            torch.manual_seed(run)
            pred_binary, _ = model.predict_with_probs(data)
            pred_binary_np = pred_binary.squeeze(0).cpu().numpy()
            onset_indices = np.where(pred_binary_np > 0)[0]

            run_onset_counts.append(len(onset_indices))
            run_onset_indices.append(onset_indices)

            print(f"Run {run}: {len(onset_indices)} onsets")

    difficulty_results[diff_name] = {
        "counts": run_onset_counts,
        "onsets": run_onset_indices,
    }

# (Optional) Summary
print("\n=== Summary ===")
for diff, res in difficulty_results.items():
    counts = res["counts"]
    print(
        f"{diff.name}: min={min(counts)}, max={max(counts)}, avg={np.mean(counts):.2f}"
    )


# Plot: Onset counts per run for each difficulty
plt.figure(figsize=(10, 5))

for i, (diff, res) in enumerate(difficulty_results.items()):
    plt.subplot(1, len(difficulty_results), i + 1)
    plt.bar(range(len(res["counts"])), res["counts"])
    plt.title(f"{diff.name} (avg: {np.mean(res['counts']):.1f})")
    plt.xlabel("Run")
    plt.ylabel("# Onsets")
    plt.ylim(0, max(max(res["counts"]) + 10, 50))  # Scale y-axis appropriately

plt.tight_layout()
plt.suptitle("Predicted Onsets Across Runs", y=1.05)


def jaccard(a, b):
    a_set, b_set = set(a), set(b)
    union = a_set | b_set
    return len(a_set & b_set) / len(union) if union else 0


fig, axes = plt.subplots(
    2, len(difficulty_results), figsize=(5 * len(difficulty_results), 8)
)
fig.suptitle("Onset Prediction Analysis", fontsize=16, y=1.02)

if len(difficulty_results) == 1:
    axes = np.expand_dims(axes, axis=1)  # Fix shape if only one difficulty

for i, (diff, res) in enumerate(difficulty_results.items()):
    counts = res["counts"]
    onsets = res["onsets"]

    # Bar chart of onset counts
    axes[0, i].bar(range(len(counts)), counts)
    axes[0, i].set_title(f"{diff.name} (avg: {np.mean(counts):.1f})")
    axes[0, i].set_xlabel("Run")
    axes[0, i].set_ylabel("# Onsets")
    axes[0, i].set_ylim(0, max(max(counts) + 10, 50))

    # Jaccard similarity matrix
    num_runs = len(onsets)
    matrix = np.zeros((num_runs, num_runs))
    for r1 in range(num_runs):
        for r2 in range(num_runs):
            matrix[r1, r2] = jaccard(onsets[r1], onsets[r2])

    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[1, i])
    avg_jaccard = np.mean(
        [matrix[r1, r2] for r1 in range(num_runs) for r2 in range(r1 + 1, num_runs)]
    )
    axes[1, i].set_title(f"{diff.name} Jaccard (avg: {avg_jaccard:.2f})")
    axes[1, i].set_xlabel("Run")
    axes[1, i].set_ylabel("Run")

plt.tight_layout()
plt.show()
