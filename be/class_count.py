import numpy as np
import os
from tqdm import tqdm
import multiprocessing as mp


def process_file(args):
    type, folder, filename = args
    file_path = os.path.join(folder, filename)
    data = np.load(file_path)
    matching_keys = [k for k in data if "_classes" in k]

    class_count = np.zeros((3, 4, 19), dtype=np.int32)
    for key in matching_keys:
        class_count += np.int32(data[key])

    return (type, len(matching_keys), class_count)


def main():
    sweep = input("Enter the sweep: ")
    difficulty = input(
        "Enter the difficulty (easy, normal, hard, expert, expertplus): "
    )
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
        args = [(type, folder, f) for f in files]

        with mp.Pool() as pool:
            for result in tqdm(
                pool.imap_unordered(process_file, args),
                total=len(files),
                desc=f"Processing {type} files",
            ):
                r_type, iter_count, class_count = result
                result_counts[f"{r_type}_iterations"] += iter_count
                result_counts[f"{r_type}_class_count"] += class_count

    np.save(f"{shuffle_folder}/class_count.npy", np.asarray(result_counts))


if __name__ == "__main__":
    main()
