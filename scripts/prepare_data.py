"""Prepare training and test data from raw signature images."""

import pickle
from pathlib import Path
from random import randrange
from typing import List

from sklearn.model_selection import train_test_split

from signature_verification.utils import fix_pair_person, fix_pair_sign

PERSON_NUMBER = 79
SIGN_NUMBER_EACH = 12


def prepare_dataset(
    data_raw_path: Path,
    output_dir: Path,
    n_samples_per_class: int = 10000,
    test_size: float = 0.05,
) -> None:
    """Prepare training and test datasets.

    Args:
        data_raw_path: Path to raw signature images directory
        output_dir: Directory to save processed dataset
        n_samples_per_class: Number of positive/negative samples to generate
        test_size: Fraction of data to use for testing
    """
    img_path_template = str(data_raw_path / "genuines" / "NFI-%03d%02d%03d.png")
    output_dir.mkdir(parents=True, exist_ok=True)

    data: List = []
    count = 0

    print(f"Generating {n_samples_per_class} pairs of each class...")

    for i in range(n_samples_per_class):
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i + 1}/{n_samples_per_class}")

        # Positive data (same person)
        anchor_person = randrange(1, PERSON_NUMBER + 1)
        anchor_sign = randrange(1, SIGN_NUMBER_EACH + 1)
        pos_sign = randrange(1, SIGN_NUMBER_EACH + 1)
        anchor_sign, pos_sign = fix_pair_sign(anchor_sign, pos_sign)

        positive = [
            img_path_template % (anchor_person, anchor_sign, anchor_person),
            img_path_template % (anchor_person, pos_sign, anchor_person),
            1,  # Label: same person
        ]

        if Path(positive[0]).exists() and Path(positive[1]).exists():
            data.append(positive)
        else:
            count += 1

        # Negative data (different persons)
        neg_person = randrange(1, PERSON_NUMBER + 1)
        neg_sign = randrange(1, SIGN_NUMBER_EACH + 1)
        anchor_person, neg_person = fix_pair_person(anchor_person, neg_person)

        negative = [
            img_path_template % (anchor_person, anchor_sign, anchor_person),
            img_path_template % (neg_person, neg_sign, neg_person),
            0,  # Label: different persons
        ]

        if Path(negative[0]).exists() and Path(negative[1]).exists():
            data.append(negative)
        else:
            count += 1

    print(f"Missing images: {count}")
    print(f"Total valid pairs: {len(data)}")

    # Split train & test
    train, test = train_test_split(data, test_size=test_size, random_state=42)

    print(f"Train samples: {len(train)}")
    print(f"Test samples: {len(test)}")

    # Save datasets
    train_path = output_dir / "train_index.pkl"
    test_path = output_dir / "test_index.pkl"

    with open(train_path, "wb") as f:
        pickle.dump(train, f)

    with open(test_path, "wb") as f:
        pickle.dump(test, f)

    print(f"Saved training data to: {train_path}")
    print(f"Saved test data to: {test_path}")


if __name__ == "__main__":
    # Paths
    data_raw = Path("./Data_raw")
    output = Path("./Data")

    if not data_raw.exists():
        print(f"Error: Data_raw directory not found at {data_raw}")
        print("Please extract Data_raw.7z first!")
        exit(1)

    prepare_dataset(data_raw, output, n_samples_per_class=10000, test_size=0.05)
    print("\n✅ Data preparation complete!")
