"""PyTorch Dataset classes for signature verification."""

import pickle
from pathlib import Path
from typing import List, Union

from torch import Tensor
from torch.utils.data import Dataset

from signature_verification.utils import convert_to_image_tensor, invert_image_path


class TrainDataset(Dataset):
    """Training dataset for signature verification.

    Loads pre-generated training pairs from pickle file.

    Args:
        data_path: Path to the pickle file containing training pairs
    """

    def __init__(self, data_path: Union[str, Path] = "./Data/train_index.pkl") -> None:
        """Initialize training dataset.

        Args:
            data_path: Path to training data pickle file
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Training data not found at {self.data_path}")

        with open(self.data_path, "rb") as train_index_file:
            self.pairs: List[List[Union[str, int]]] = pickle.load(train_index_file)

    def __getitem__(self, index: int) -> List[Union[Tensor, int]]:
        """Get a training pair.

        Args:
            index: Index of the pair

        Returns:
            List containing [image1_tensor, image2_tensor, label]
        """
        item = self.pairs[index]
        x = convert_to_image_tensor(invert_image_path(item[0]))
        y = convert_to_image_tensor(invert_image_path(item[1]))
        return [x, y, item[2]]

    def __len__(self) -> int:
        """Return the number of pairs in the dataset.

        Returns:
            Number of training pairs
        """
        return len(self.pairs)


class SignatureTestDataset(Dataset):
    """Test dataset for signature verification.

    Loads pre-generated test pairs from pickle file.

    Args:
        data_path: Path to the pickle file containing test pairs
    """

    def __init__(self, data_path: Union[str, Path] = "./Data/test_index.pkl") -> None:
        """Initialize test dataset.

        Args:
            data_path: Path to test data pickle file
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Test data not found at {self.data_path}")

        with open(self.data_path, "rb") as test_index_file:
            self.pairs: List[List[Union[str, int]]] = pickle.load(test_index_file)

    def __getitem__(self, index: int) -> List[Union[Tensor, int]]:
        """Get a test pair.

        Args:
            index: Index of the pair

        Returns:
            List containing [image1_tensor, image2_tensor, label]
        """
        item = self.pairs[index]
        x = convert_to_image_tensor(invert_image_path(item[0]))
        y = convert_to_image_tensor(invert_image_path(item[1]))
        return [x, y, item[2]]

    def __len__(self) -> int:
        """Return the number of pairs in the dataset.

        Returns:
            Number of test pairs
        """
        return len(self.pairs)
