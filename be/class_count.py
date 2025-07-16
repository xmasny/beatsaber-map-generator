import numpy as np
import os
from tqdm import tqdm
import multiprocessing as mp


def worker(task_queue, result_queue, position, min_keys_for_bar=100):
    while True:
        task = task_queue.get()
        if task is None:
            break

        type, folder, filename = task
        file_path = os.path.join(folder, filename)

        try:
            data = np.load(file_path)
        except Exception as e:
            print(f"[ERROR] {file_path}: {e}")
            result_queue.put((type, 0, np.zeros((3, 4, 19), dtype=np.int32)))
            continue

        matching_keys = [k for k in data if "_classes" in k]
        class_count = np.zeros((3, 4, 19), dtype=np.int32)

        if len(matching_keys) >= min_keys_for_bar:
            bar = tqdm(matching_keys, desc=filename, position=position + 1, leave=False)
        else:
            print(f"[{type.upper()}] {filename}: {len(matching_keys)} keys")
            bar = matching_keys  # just a regular iterable

        for key in bar:
            try:
                class_count += np.int32(data[key])
            except Exception as e:
                print(f"[WARN] {filename}:{key}: {e}")

        if isinstance(bar, tqdm):
            bar.close()
            print(f"[{type.upper()}] {filename}: {len(matching_keys)} keys processed")

        result_queue.put((type, len(matching_keys), class_count))


def main():
    difficulty = input(
        "Enter the difficulty (easy, normal, hard, expert, expertplus): "
    )
    seeds = os.listdir(f"dataset/batch/{difficulty}")
    print("Available seeds:", seeds)
    sweep = input("Enter the sweep: ")
    shuffle_folder = f"dataset/batch/{difficulty}/{sweep}"

    result_counts = {
        "train_class_count": np.zeros((3, 4, 19), dtype=np.int32),
        "train_iterations": 0,
        "validation_class_count": np.zeros((3, 4, 19), dtype=np.int32),
        "validation_iterations": 0,
        "test_class_count": np.zeros((3, 4, 19), dtype=np.int32),
        "test_iterations": 0,
    }

    for type in ["train", "validation", "test"]:
        folder = os.path.join(shuffle_folder, type, "class")
        files = os.listdir(folder)
        tasks = [(type, folder, f) for f in files]

        task_queue = mp.Queue()
        result_queue = mp.Queue()
        num_workers = min(8, len(files))  # tune based on CPU
        workers = []

        print(f"\n=== STARTING {type.upper()} SET ({len(files)} files) ===")

        for i in range(num_workers):
            p = mp.Process(target=worker, args=(task_queue, result_queue, i))
            p.start()
            workers.append(p)

        for task in tasks:
            task_queue.put(task)
        for _ in range(num_workers):
            task_queue.put(None)  # poison pills

        with tqdm(
            total=len(tasks),
            desc=f"{type.capitalize()} Files",
            position=0,
            dynamic_ncols=True,
        ) as file_bar:
            for _ in range(len(tasks)):
                r_type, iter_count, class_count = result_queue.get()
                result_counts[f"{r_type}_iterations"] += iter_count
                result_counts[f"{r_type}_class_count"] += class_count
                file_bar.update(1)
        for p in workers:
            p.join()

        print(f"=== FINISHED {type.upper()} SET ===\n")

    out_path = f"{shuffle_folder}/class_count.npy"
    np.save(out_path, np.asarray(result_counts))
    print(f"Saved class counts to {out_path}")


if __name__ == "__main__":
    main()
