import numpy as np
import pandas as pd
from tqdm import tqdm

difficulty = input("Enter difficulty: ")

df = pd.read_csv(
    f"dataset/color_notes/{difficulty}.csv",
    header=None,
).values.tolist()

corrupted = []

pbar = tqdm(df, total=len(df))

for i, song in enumerate(pbar):
    try:
        with open(f"dataset/color_notes/{difficulty}/{song[0]}", "rb") as f:
            np.load(f)

        with open(f"dataset/songs/{song[0]}", "rb") as f:
            np.load(f)

    except Exception as e:
        corrupted.append(i)
        print(e, song[0])
        continue

corrupted.reverse()

for i in corrupted:
    df.pop(i)

print(f"New length: {len(df)}")

df = pd.DataFrame(df)
df.to_csv(f"dataset/{difficulty}_new.csv", index=False, header=False)
