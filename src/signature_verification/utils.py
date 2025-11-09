"""Utility functions for signature preprocessing and data generation."""

from pathlib import Path
from random import randrange
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.ImageOps import invert
from torch import Tensor

PERSON_NUMBER = 79
SIGN_NUMBER_EACH = 12


def fix_pair_person(x: int, y: int) -> Tuple[int, int]:
    """Ensure two person IDs are different.

    Recursively generates new person IDs until they differ.

    Args:
        x: First person ID
        y: Second person ID

    Returns:
        Tuple of two different person IDs
    """
    if x == y:
        return fix_pair_person(x, randrange(1, PERSON_NUMBER + 1))
    return x, y


def fix_pair_sign(x: int, y: int) -> Tuple[int, int]:
    """Ensure two signature IDs are different.

    Recursively generates new signature IDs until they differ.

    Args:
        x: First signature ID
        y: Second signature ID

    Returns:
        Tuple of two different signature IDs
    """
    if x == y:
        return fix_pair_sign(x, randrange(1, SIGN_NUMBER_EACH + 1))
    return x, y


def invert_image_path(path: Union[str, Path]) -> NDArray[np.uint8]:
    """Load and preprocess a signature image from path.

    Steps:
        1. Load image and convert to grayscale
        2. Resize to 220x155
        3. Invert colors
        4. Apply binary threshold (50)

    Args:
        path: Path to the signature image file

    Returns:
        Preprocessed image array of shape (155, 220)

    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image_file = Image.open(path)
    image_file = image_file.convert("L").resize([220, 155])
    image_file = invert(image_file)
    image_array = np.array(image_file, dtype=np.uint8)
    image_array[image_array >= 50] = 255
    image_array[image_array < 50] = 0
    return image_array


def convert_to_image_tensor(image_array: NDArray[np.uint8]) -> Tensor:
    """Convert preprocessed image array to PyTorch tensor.

    Normalizes pixel values to [0, 1] and reshapes for CNN input.

    Args:
        image_array: Preprocessed image array of shape (155, 220)

    Returns:
        Normalized tensor of shape (1, 220, 155)
    """
    normalized = image_array.astype(np.float32) / 255.0
    tensor = Tensor(normalized)
    return tensor.view(1, 220, 155)


def show_inverted(path: Union[str, Path]) -> None:
    """Display inverted and preprocessed signature image.

    Args:
        path: Path to the signature image file
    """
    img = Image.fromarray(invert_image_path(path))
    img.show()


def invert_image(image_file: Image.Image) -> NDArray[np.uint8]:
    """Invert and preprocess a PIL Image.

    Alternative to invert_image_path that works with Image objects.

    Args:
        image_file: PIL Image object

    Returns:
        Preprocessed image array of shape (155, 220)
    """
    image_file = image_file.convert("L").resize([220, 155])
    image_file = invert(image_file)
    image_array = np.array(image_file, dtype=np.uint8)
    image_array[image_array >= 50] = 255
    image_array[image_array < 50] = 0
    return image_array
