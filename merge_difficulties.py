import pandas as pd


def aggregate_booleans(series):
    return series.any()


difficulties = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]

for difficulty in difficulties:
    df = pd.read_csv(
        f"dataset/beatmaps/color_notes/{difficulty}.csv",
        names=[
            "song",
            "upvotes",
            "downvotes",
            "score",
            "bpm",
            "duration",
            "automapper",
        ],
    )
    # df.drop(df.columns[1], axis=1, inplace=True)
    df[difficulty] = True

    df.to_csv(f"dataset/beatmaps/color_notes/{difficulty}.csv", index=False)


merge = pd.concat(
    [
        pd.read_csv(f"dataset/beatmaps/color_notes/{difficulty}.csv")
        for difficulty in difficulties
    ]
)

merge.to_csv("dataset/beatmaps/color_notes/merged.csv", index=False)

df = pd.read_csv("dataset/beatmaps/color_notes/merged.csv")

combined_df = (
    df.groupby(
        ["song", "upvotes", "downvotes", "score", "bpm", "duration", "automapper"]
    )
    .agg(
        {
            "Easy": aggregate_booleans,
            "Normal": aggregate_booleans,
            "Hard": aggregate_booleans,
            "Expert": aggregate_booleans,
            "ExpertPlus": aggregate_booleans,
        }
    )
    .reset_index()
)

# Save to a new CSV file (optional)
combined_df.to_csv("dataset/beatmaps/color_notes/combined_songs.csv", index=False)
