import json
import matplotlib.pyplot as plt


def plot_batch_size_search(search_history):
    batch_sizes = [
        entry[0] for entry in search_history if entry[1] and entry[2] is not None
    ]
    times = [entry[2] for entry in search_history if entry[1] and entry[2] is not None]

    if not batch_sizes:
        print("❌ No successful batch size search results found.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, times, marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Avg Time per Batch (s)")
    plt.title("Batch Size vs Avg Loading + Forward Time")
    plt.grid(True)
    plt.show()


def plot_worker_benchmark(worker_benchmarks):
    workers = [entry[0] for entry in worker_benchmarks]
    times = [entry[1] for entry in worker_benchmarks]

    if not workers:
        print("❌ No worker benchmark results found.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(workers, times, marker="o", color="green")
    plt.xlabel("Number of Workers")
    plt.ylabel("Avg Time per Batch (s)")
    plt.title("Workers vs Avg Loading + Forward Time")
    plt.grid(True)
    plt.show()


def plot_epoch_times(epoch_data):
    epochs = list(range(1, len(epoch_data["epoch_times"]) + 1))
    times = epoch_data["epoch_times"]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, times, marker="o", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Time per Epoch (s)")
    plt.title(
        f"Epoch Times | Batch Size {epoch_data['batch_size']}, Workers {epoch_data['num_workers']}"
    )
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("🛎 Loading tuning results...")

    # Try loading auto_tune_results.json
    try:
        with open("auto_tune_results.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
        print("⚠️ Warning: Cannot find auto_tune_results.json!")

    # Try loading epoch_times.json
    try:
        with open("epoch_times.json", "r") as f:
            epoch_data = json.load(f)
    except FileNotFoundError:
        epoch_data = None
        print("⚠️ Warning: Cannot find epoch_times.json!")

    print("\nAvailable plots:")
    print("1. Batch size search results")
    print("2. Worker benchmarks")
    print("3. Both batch+worker benchmarks")
    if epoch_data:
        print("4. Epoch times (full train+validate benchmark)")

    choice = input("Select [1/2/3/4]: ").strip()

    if choice == "1" or choice == "3":
        if "search_history" in data:
            plot_batch_size_search(data["search_history"])
        else:
            print("❌ No batch size search data available.")

    if choice == "2" or choice == "3":
        if "worker_benchmarks" in data:
            plot_worker_benchmark(data["worker_benchmarks"])
        else:
            print("❌ No worker benchmark data available.")

    if choice == "4":
        if epoch_data:
            plot_epoch_times(epoch_data)
        else:
            print("❌ No epoch timing data available.")
