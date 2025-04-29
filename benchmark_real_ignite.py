import json
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from training.loader import BaseLoader
from dl.models.onsets import SimpleOnsets
from training.onset_ignite import ignite_train
from config import *
from utils import MyDataParallel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== TUNING PARAMETERS ====
START_BATCH = 8
MAX_BATCH = 512
MIN_BATCH = 16
SHRINK_FACTOR = 0.9
MAX_TIME_PER_BATCH = 5
MAX_TOTAL_TIME = 30
BATCHES_TO_TEST = 5
EPOCHS = 2
WORKER_OPTIONS = [8, 6, 4, 2, 0]
IS_PARALLEL = True

# ==== RESULTS ====
benchmark_file = "real_ignite_benchmark.json"
config_file = "training_config.json"
results = []


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


def test_batch_loading(batch_size, num_workers):
    try:
        dataset = BaseLoader(
            min_sum_votes=0,
            min_score=0,
            min_bpm=60,
            max_bpm=300,
            difficulty=DifficultyName.ALL,
            object_type=ObjectType.COLOR_NOTES,
            enable_condition=True,
            with_beats=True,
            seq_length=16000,
            skip_step=2000,
            valid_split_ratio=0.2,
            skip_processing=False,
            split=Split.TRAIN,
            split_seed=42,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        loader = dataset.get_dataloader()
        model = setup_model()
        model.eval()

        it = iter(loader)
        start = time.time()
        for _ in range(BATCHES_TO_TEST):
            batch = next(it)
            for key in batch:
                batch[key] = batch[key].to(device)
            with torch.no_grad():
                model.run_on_batch(batch)
            if time.time() - start > MAX_TOTAL_TIME:
                return None
        elapsed = time.time() - start
        return elapsed / BATCHES_TO_TEST
    except Exception:
        torch.cuda.empty_cache()
        return None


def binary_search_batch(num_workers):
    print(f"🔍 Starting binary search with {num_workers} workers")
    low, high = START_BATCH, MAX_BATCH
    best = None

    while low <= high:
        mid = (low + high) // 2
        avg_time = test_batch_loading(mid, num_workers)
        if avg_time is not None:
            print(f"✅ Batch {mid} ok ({avg_time:.2f}s/batch)")
            best = mid
            low = mid + 1
        else:
            print(f"❌ Batch {mid} failed")
            high = mid - 1

    return best


def auto_shrink(batch_size, num_workers):
    while batch_size >= MIN_BATCH:
        print(f"🔄 Shrinking test at batch_size={batch_size}")
        if test_batch_loading(batch_size, num_workers) is not None:
            return batch_size
        batch_size = int(batch_size * SHRINK_FACTOR)
    return None


# ==== Start ====
print("🚀 Tuning batch size...")
best_batch_size = binary_search_batch(num_workers=WORKER_OPTIONS[0])
best_batch_size = auto_shrink(best_batch_size, WORKER_OPTIONS[0])

if best_batch_size is None:
    print("❌ No valid batch size found.")
    exit(1)

print(f"🎯 Final batch size: {best_batch_size}")
print("⚙️ Benchmarking real ignite_train() on different num_workers...")

best_result = None
for workers in WORKER_OPTIONS:
    try:
        train_dataset = BaseLoader(
            split=Split.TRAIN, num_workers=workers, batch_size=best_batch_size
        )
        valid_dataset = BaseLoader(
            split=Split.VALIDATION, num_workers=workers, batch_size=best_batch_size
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
        print(f"✅ Workers {workers}: {avg_time:.2f}s/epoch")

        result = {
            "batch_size": best_batch_size,
            "num_workers": workers,
            "avg_epoch_time": avg_time,
            "total_time": total_time,
        }
        results.append(result)

        if not best_result or avg_time < best_result["avg_epoch_time"]:
            best_result = result

    except Exception as e:
        print(f"❌ Failed with workers={workers}: {str(e)}")
        torch.cuda.empty_cache()
        results.append(
            {
                "batch_size": best_batch_size,
                "num_workers": workers,
                "avg_epoch_time": None,
                "error": str(e),
            }
        )

# ==== Save Results ====
with open("real_ignite_benchmark.json", "w") as f:
    json.dump(results, f, indent=4)

if best_result:
    with open("training_config.json", "w") as f:
        json.dump(
            {
                "batch_size": best_result["batch_size"],
                "num_workers": best_result["num_workers"],
            },
            f,
            indent=4,
        )
    print(
        f"\n🏆 Best config: batch={best_result['batch_size']}, workers={best_result['num_workers']}, {best_result['avg_epoch_time']:.2f}s/epoch"
    )
else:
    print("\n❌ No valid training configuration found.")
