import json
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from training.onset_loader import BaseLoader
from dl.models.onsets import OnsetFeatureExtractor
from training.onset_ignite import ignite_train
from config import *
from utils import MyDataParallel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu", 1)

# Configuration
MIN_BATCH_SIZE = 16
MAX_BATCH_SIZE = 64
EPOCHS = 5
PROBE_OFFSET = 5  # How many sizes below max to test
benchmark_file = "real_ignite_benchmark.json"
config_file = "training_config.json"

IS_PARALLEL = False  # Set to True if using multiple GPUs


results = []
best_result = None


def setup_model():
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
    if IS_PARALLEL:
        model = MyDataParallel(model)
    return model


def test_batch_run(batch_size):
    try:
        train_dataset = BaseLoader(split=Split.TRAIN, batch_size=batch_size)
        valid_dataset = BaseLoader(split=Split.VALIDATION, batch_size=batch_size)

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

        avg_time = (end - start) / EPOCHS
        print(f"✅ batch_size={batch_size} → {avg_time:.2f}s/epoch")
        return avg_time

    except Exception as e:
        print(f"❌ batch_size={batch_size} → {e}")
        torch.cuda.empty_cache()
        return None


if __name__ == "__main__":

    # Main benchmark loop
    print("🚀 Starting max-batch + timing probe benchmark...")

    low = MIN_BATCH_SIZE
    high = MAX_BATCH_SIZE
    best_batch = None

    while low <= high:
        mid = (low + high) // 2
        success = test_batch_run(mid)
        if success is not None:
            best_batch = mid
            low = mid + 1
        else:
            high = mid - 1

    if best_batch is None:
        print("⚠️ No working batch size found")
        exit(1)

    print(f"🎯 Max working batch size is {best_batch}")

    # Test max and a few smaller batch sizes
    for probe_batch in range(
        best_batch, max(best_batch - PROBE_OFFSET, MIN_BATCH_SIZE - 1), -1
    ):
        avg_time = test_batch_run(probe_batch)
        result = {
            "batch_size": probe_batch,
            "avg_epoch_time": avg_time,
        }
        results.append(result)

        if avg_time is not None and (
            best_result is None or avg_time < best_result["avg_epoch_time"]
        ):
            best_result = result

    # Save all results
    with open(benchmark_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"📄 Benchmark results saved to {benchmark_file}")

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
            f"   → batch={best_result['batch_size']}, workers={best_result['num_workers']}, time={best_result['avg_epoch_time']:.2f}s/epoch"
        )
    else:
        print("❌ No working configuration found.")
