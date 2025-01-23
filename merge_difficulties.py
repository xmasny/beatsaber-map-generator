import pandas as pd


def aggregate_booleans(series):
    return series.any()


difficulties = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
types = ["color_notes", "obstacles", "bomb_notes"]

for object_type in types:
    merge = pd.concat(
        [
            pd.read_csv(f"dataset/beatmaps/{object_type}/{difficulty}.csv")
            for difficulty in difficulties
        ]
    )

    merge.to_csv(f"dataset/beatmaps/{object_type}/merged.csv", index=False)

    print(f"Merged {object_type} datasets")

    df = pd.read_csv(f"dataset/beatmaps/{object_type}/merged.csv")

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
    combined_df.to_csv(f"dataset/beatmaps/{object_type}/combined_songs.csv", index=False)

    print(f"Combined {object_type} datasets")
