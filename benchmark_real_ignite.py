import json
import sys
import time
import torch.multiprocessing as mp

if sys.platform.startswith("linux"):
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Start method was already set; that's OK
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from training.loader import BaseLoader
from dl.models.onsets import SimpleOnsets
from training.onset_ignite import ignite_train
from config import *
from utils import MyDataParallel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
MAX_BATCH_SIZE = 512
MIN_BATCH_SIZE = 32
SHRINK_FACTOR = 0.9
WORKER_OPTIONS = [8, 6, 4, 2]
EPOCHS = 2
IS_PARALLEL = True

results = []
benchmark_file = "real_ignite_benchmark.json"
config_file = "training_config.json"


def setup_model():
    model = SimpleOnsets(
        input_features=n_mels,
        output_features=1,
        dropout=0.5,
        rnn_dropout=0.1,
        enable_condition=True,
        num_layers=2,
        enable_beats=True,
        inference_chunk_length=round(16000 / FRAME),
    ).to(device)

    if IS_PARALLEL:
        model = MyDataParallel(model)
    return model


# Benchmark
print("🚀 Starting real ignite_train benchmark (shrink on fail)...")

best_result = None
for num_workers in WORKER_OPTIONS:
    batch_size = MAX_BATCH_SIZE
    while batch_size >= MIN_BATCH_SIZE:
        print(f"🔧 Trying: batch_size={batch_size}, workers={num_workers}")
        try:
            train_dataset = BaseLoader(
                split=Split.TRAIN, num_workers=num_workers, batch_size=batch_size
            )
            valid_dataset = BaseLoader(
                split=Split.VALIDATION, num_workers=num_workers, batch_size=batch_size
            )

            train_loader = train_dataset.get_dataloader()
            valid_loader = valid_dataset.get_dataloader()
            model = setup_model()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            lr_scheduler = CyclicLR(
                optimizer,
                base_lr=0.000493,
                max_lr=0.00241,
                step_size_up=1000,
                cycle_momentum=False,
            )

            start = time.time()
            ignite_train(
                train_dataset,
                valid_dataset,
                model,  # type: ignore
                train_loader,
                valid_loader,
                optimizer,
                len(train_dataset),
                len(valid_dataset),
                device,
                lr_scheduler,
                wandb_logger=None,
                epochs=EPOCHS,
                wandb_mode="disabled",
            )
            end = time.time()

            total_time = end - start
            avg_time = total_time / EPOCHS
            print(f"✅ Success: {avg_time:.2f}s/epoch")

            result = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "avg_epoch_time": avg_time,
                "total_time": total_time,
            }
            results.append(result)

            if not best_result or avg_time < best_result["avg_epoch_time"]:
                best_result = result

                break  # success, stop shrinking for this worker setting

        except Exception as e:
            print(f"❌ Failed: batch={batch_size}, workers={num_workers} → {e}")
            torch.cuda.empty_cache()
            batch_size = int(batch_size * SHRINK_FACTOR)

# Save all results
with open(benchmark_file, "w") as f:
    json.dump(results, f, indent=4)
print(f"📄 All benchmark results saved to {benchmark_file}")

# Save best config
if best_result:
    with open(config_file, "w") as f:
        json.dump(
            {
                "batch_size": best_result["batch_size"],
                "num_workers": best_result["num_workers"],
            },
            f,
            indent=4,
        )
    print(f"🏆 Best config saved to {config_file}")
    print(
        f"   → batch={best_result['batch_size']}, workers={best_result['num_workers']}, {best_result['avg_epoch_time']:.2f}s/epoch"
    )
else:
    print("❌ No working configuration found.")
