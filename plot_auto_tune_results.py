# plot_auto_tune_results.py
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


if __name__ == "__main__":
    print("🛎 Loading tuning results...")
    try:
        with open("auto_tune_results.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Cannot find auto_tune_results.json!")
        exit(1)

    print("\nAvailable plots:")
    print("1. Batch size search results")
    print("2. Worker benchmarks")
    print("3. Both")

    choice = input("Select [1/2/3]: ").strip()

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
