"""Download the AI4I 2020 predictive maintenance dataset."""

from pathlib import Path
from urllib.request import urlretrieve

from preprocess import DATA_PATH

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        print(f"Dataset already exists: {DATA_PATH}")
        return

    print(f"Downloading dataset from {DATA_URL}")
    urlretrieve(DATA_URL, DATA_PATH)
    print(f"Saved dataset to {DATA_PATH}")


if __name__ == "__main__":
    main()
