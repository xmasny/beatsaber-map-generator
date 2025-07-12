from collections import defaultdict
from itertools import count
from threading import Lock, Thread
from queue import Queue
from tqdm import tqdm
import os
import time
import requests

base_dataset_path = "http://kaistore.dcs.fmph.uniba.sk/beatsaber-map-generator/"


class Downloader:
    def __init__(self, download_fn, max_queue_size=100, position_start=2):
        self.download_fn = download_fn
        self.queue = Queue(maxsize=max_queue_size)
        self.status = defaultdict(lambda: "not_started")
        self.progress = {}
        self.lock = Lock()
        self.thread = Thread(target=self.worker)
        self.bar_position = position_start
        self.thread.daemon = True
        self.thread.start()

    def worker(self):
        while True:
            rel_path, save_path = self.queue.get()
            with self.lock:
                self.status[rel_path] = "downloading"
                self.progress[rel_path] = (0, 0)

            bar_position = self.bar_position

            # Create tqdm progress bar
            bar = tqdm(
                desc=f"Downloading {rel_path.split('/')[-3]}/{rel_path.split('/')[-1]}",
                total=1.0,
                unit="B",
                unit_scale=True,
                position=bar_position,
                leave=True,
            )

            def progress_callback(downloaded, total):
                with self.lock:
                    self.progress[rel_path] = (downloaded, total)
                bar.total = total
                bar.n = downloaded
                bar.refresh()

            try:
                self.download_fn(rel_path, save_path, progress_callback)
                with self.lock:
                    self.status[rel_path] = "done"
            except Exception as e:
                print(f"[Download error] {rel_path}: {e}")
                with self.lock:
                    self.status[rel_path] = "failed"
            finally:
                bar.close()
                self.queue.task_done()

    def enqueue(self, rel_path, save_path):
        if not os.path.exists(save_path) and self.status[rel_path] == "not_started":
            self.queue.put((rel_path, save_path))

    def get_progress(self, rel_path):
        with self.lock:
            return self.progress.get(rel_path, (0, 1))

    def get_status(self, rel_path):
        with self.lock:
            return self.status[rel_path]


def download_fn(rel_path, save_path, progress_callback=None):
    url = f"{base_dataset_path}/{rel_path}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    temp_path = save_path + ".part"

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0

        with open(temp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total)
    os.rename(temp_path, save_path)
