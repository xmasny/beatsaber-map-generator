# auto_tune_training.py

import os
import sys
import torch.multiprocessing as mp
import wandb

if sys.platform.startswith("linux"):
    mp.set_start_method("spawn", force=True)

print(f"🔵 Multiprocessing start method: {mp.get_start_method()}")

import json
import time
import torch
import threading
from typing import Optional
from training.loader import BaseLoader
from config import *
from dl.models.onsets import SimpleOnsets
from tqdm import tqdm
from training.onset_ignite import ignite_train
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR

from utils import MyDataParallel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Constants ---
START_BATCH = 8
MAX_SEARCH_BATCH = 1024  # For local; change to 1024 for server
DEFAULT_NUM_WORKERS = 8  # For local tuning
BATCHES_TO_TEST = 10
MAX_TOTAL_TIME = 30  # seconds
MAX_TIME_PER_BATCH = 5  # seconds
MIN_BATCH_SIZE = 16
SHRINK_FACTOR = 0.9
IS_PARALLEL = True  # Set to True if using multiple GPUs

# --- Global flags ---
search_history = []
worker_benchmarks = []
user_requested_stop = False


def keyboard_listener():
    global user_requested_stop
    while True:
        key = sys.stdin.read(1)
        if key.lower() == "s":
            print("\n🔴 Detected 's' key — stopping gracefully...")
            user_requested_stop = True
            break


def cycle(iteration, train_dataset: BaseLoader, num_songs_pbar: Optional[tqdm] = None):
    if num_songs_pbar:
        num_songs_pbar.reset()
    for index, songs in enumerate(iteration):
        for song in songs:
            if num_songs_pbar:
                num_songs_pbar.update(1)
            if "not_working" in song:
                continue
            for segment in train_dataset.process(song_meta=song):
                yield segment


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
    model.eval()
    return model


def test_batch(batch_size, num_workers, batches_to_test=BATCHES_TO_TEST):
    try:
        train_dataset = BaseLoader(
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

        train_loader = train_dataset.get_dataloader()
        model = setup_model()

        iteration = iter(train_loader)
        song_batches = cycle(iteration, train_dataset)

        pbar = tqdm(
            total=batches_to_test,
            desc=f"BatchSize {batch_size} | Workers {num_workers}",
            leave=False,
        )

        start_total_time = time.time()

        for _ in range(batches_to_test):
            if user_requested_stop:
                pbar.close()
                return None

            batch_start = time.time()

            batch = next(song_batches)

            for key in batch:
                batch[key] = batch[key].to(device, non_blocking=True)

            with torch.no_grad():
                outputs, losses = model.run_on_batch(batch)

            batch_end = time.time()
            pbar.update(1)

            if batch_end - batch_start > MAX_TIME_PER_BATCH:
                print(
                    f"\n⏳ Single batch too slow ({batch_end - batch_start:.2f}s), skipping batch_size={batch_size}"
                )
                pbar.close()
                return None

            if batch_end - start_total_time > MAX_TOTAL_TIME:
                print(
                    f"\n⏳ Total loading too slow ({batch_end - start_total_time:.2f}s), skipping batch_size={batch_size}"
                )
                pbar.close()
                return None

        pbar.close()

        end_total_time = time.time()
        elapsed_time = end_total_time - start_total_time

        del train_dataset, train_loader, iteration, batch, model
        torch.cuda.empty_cache()

        return elapsed_time / batches_to_test  # avg time per batch

    except RuntimeError as e:
        torch.cuda.empty_cache()
        return None


def binary_search_max_batch(
    start_batch=START_BATCH,
    max_search=MAX_SEARCH_BATCH,
    num_workers=DEFAULT_NUM_WORKERS,
):
    global user_requested_stop
    low = start_batch
    high = max_search
    best = start_batch

    while low <= high:
        if user_requested_stop:
            break

        mid = (low + high) // 2
        print(f"🚀 Testing batch_size={mid}...")
        avg_time = test_batch(mid, num_workers)

        if avg_time is not None:
            best = mid
            low = mid + 1
            search_history.append((mid, True, avg_time))
            print(f"✅ batch_size {mid} succeeded ({avg_time:.4f} sec/batch)")
        else:
            high = mid - 1
            search_history.append((mid, False, None))
            print(f"❌ batch_size {mid} failed")

    return best


def auto_shrink_and_benchmark(best_batch_size, worker_options=[2, 4, 8, 12, 16]):
    batch_size = best_batch_size

    while batch_size >= MIN_BATCH_SIZE:
        print(f"\n🔵 Benchmarking workers at batch_size={batch_size}...")
        current_worker_benchmarks = []

        for workers in worker_options:
            print(f"Testing {workers} workers...")
            avg_time = test_batch(batch_size, workers)

            if avg_time is not None:
                current_worker_benchmarks.append((workers, avg_time))
                print(f"✅ {workers} workers: {avg_time:.4f} sec/batch")
            else:
                print(f"❌ {workers} workers failed")

        if current_worker_benchmarks:
            worker_benchmarks.clear()
            worker_benchmarks.extend(current_worker_benchmarks)
            print(f"\n🏆 Found working batch_size={batch_size}!")
            return batch_size

        else:
            print(f"\n⚡ Batch_size={batch_size} failed for all workers, shrinking...")
            batch_size = int(batch_size * SHRINK_FACTOR)

    print("\n❌ Could not find a working batch size above minimum.")
    return None


if __name__ == "__main__":
    print("🛎 Auto-tune training configuration...")
    print("🔵 Press 's' anytime to stop.")

    print("\nChoose tuning mode:")
    print("1. Search batch size only")
    print("2. Benchmark num_workers only")
    print("3. Full search (batch size + workers)")
    print("4. Fault-tolerant benchmark (batch size shrinking + workers fallback)")
    mode = input("Select mode [1/2/3/4]: ").strip()

    listener = threading.Thread(target=keyboard_listener, daemon=True)
    listener.start()

    if mode == "1":
        best_batch_size = binary_search_max_batch()
        print(f"\n🎯 Best batch size: {best_batch_size}")

        with open("auto_tune_results.json", "w") as f:
            json.dump(
                {"best_batch_size": best_batch_size, "search_history": search_history},
                f,
            )

    elif mode == "2":
        try:
            with open("auto_tune_results.json", "r") as f:
                results = json.load(f)
                best_batch_size = results["best_batch_size"]
        except FileNotFoundError:
            print("❌ No batch size found. Please run batch size search first.")
            exit(1)

        best_batch_size = auto_shrink_and_benchmark(best_batch_size)
        if best_batch_size is None:
            print("❌ No valid batch size found after shrinking.")
            exit(1)

        with open("auto_tune_results.json", "r+") as f:
            data = json.load(f)
            data["worker_benchmarks"] = worker_benchmarks
            f.seek(0)
            json.dump(data, f)
            f.truncate()

    elif mode == "3":
        best_batch_size = binary_search_max_batch()
        print(f"\n🎯 Best batch size: {best_batch_size}")

        best_batch_size = auto_shrink_and_benchmark(best_batch_size)
        if best_batch_size is None:
            print("❌ No valid batch size found after shrinking.")
            exit(1)

        with open("auto_tune_results.json", "w") as f:
            json.dump(
                {
                    "best_batch_size": best_batch_size,
                    "search_history": search_history,
                    "worker_benchmarks": worker_benchmarks,
                },
                f,
            )
    elif mode == "4":
        best_batch_size = None
        print(
            "\n🚀 Running fault-tolerant benchmark (batch size shrinking + workers fallback)..."
        )

        if os.path.exists("auto_tune_results.json"):
            print("📂 Found existing auto_tune_results.json. Skipping batch search...")
            with open("auto_tune_results.json", "r") as f:
                config = json.load(f)
                best_batch_size = config["best_batch_size"]
        else:
            best_batch_size = binary_search_max_batch()
            print(f"🎯 Initial best batch size: {best_batch_size}")

            best_batch_size = auto_shrink_and_benchmark(best_batch_size)
        if best_batch_size is None:
            print("❌ No valid batch size found after shrinking.")
            exit(1)

        shrink_step = max(16, int(best_batch_size * 0.1))
        min_batch_size = 16
        epochs = 3
        success = False

        worker_options = [8, 6, 4, 2]

        for num_workers in worker_options:
            batch_size = best_batch_size
            while batch_size >= min_batch_size:
                try:
                    print(f"🔧 Trying batch_size={batch_size}, workers={num_workers}")

                    train_dataset = BaseLoader(
                        split=Split.TRAIN,
                        num_workers=num_workers,
                        batch_size=batch_size,
                    )

                    valid_dataset = BaseLoader(
                        split=Split.VALIDATION,
                        num_workers=num_workers,
                        batch_size=batch_size,
                    )

                    train_loader = train_dataset.get_dataloader()
                    valid_loader = valid_dataset.get_dataloader()
                    train_dataset_len = len(train_dataset)
                    valid_dataset_len = len(valid_dataset)

                    model = SimpleOnsets(
                        input_features=n_mels,
                        output_features=1,
                        dropout=0.4,
                        rnn_dropout=0.1,
                        num_layers=2,
                        inference_chunk_length=round(16000 / FRAME),
                    ).to(device)

                    optimizer = optim.Adam(model.parameters(), lr=1e-3)
                    lr_scheduler = CyclicLR(
                        optimizer,
                        base_lr=0.000493,
                        max_lr=0.00241,
                        step_size_up=1000,
                        cycle_momentum=False,
                    )

                    print("\n⏱ Running ignite_train() for 3 epochs...")
                    start = time.time()

                    run_parameters = {"epochs": epochs, "wandb_mode": "disabled"}

                    ignite_train(
                        train_dataset,
                        valid_dataset,
                        model,
                        train_loader,
                        valid_loader,
                        optimizer,
                        train_dataset_len,
                        valid_dataset_len,
                        device,
                        lr_scheduler,
                        epochs=epochs,
                        wandb_logger=None,
                        **run_parameters,
                    )

                    end = time.time()
                    total_time = end - start
                    avg_epoch_time = total_time / epochs

                    print(f"✅ Total training time: {total_time:.2f}s")
                    print(f"📊 Average epoch time: {avg_epoch_time:.2f}s")

                    epoch_results = {
                        "batch_size": batch_size,
                        "num_workers": num_workers,
                        "epoch_times": [avg_epoch_time] * epochs,
                        "average_epoch_time": avg_epoch_time,
                    }

                    with open("epoch_times.json", "w") as f:
                        json.dump(epoch_results, f, indent=4)

                    with open("training_config.json", "w") as f:
                        json.dump(
                            {"batch_size": batch_size, "num_workers": num_workers},
                            f,
                            indent=4,
                        )

                    print(
                        "📄 Results saved to epoch_times.json and training_config.json"
                    )
                    success = True
                    break

                except RuntimeError as e:
                    print(
                        f"❌ Failed at batch_size={batch_size}, workers={num_workers}: {e}"
                    )
                    batch_size -= shrink_step
                    torch.cuda.empty_cache()
                    continue

            if success:
                break

        if not success:
            print("❌ Could not find any working combination. Training failed.")

    else:
        print("❌ Invalid mode selected.")
        exit(1)

    print("\n📦 Results saved to auto_tune_results.json")
    print("📊 You can now run 'plot_auto_tune_results.py' to visualize the results!")
