"""Tests for utility functions."""

import numpy as np
import pytest
import torch
from PIL import Image

from signature_verification.utils import (
    PERSON_NUMBER,
    SIGN_NUMBER_EACH,
    convert_to_image_tensor,
    fix_pair_person,
    fix_pair_sign,
    invert_image,
)


class TestFixPairs:
    """Test suite for pair fixing functions."""

    def test_fix_pair_person_different(self):
        """Test that fix_pair_person returns different IDs."""
        x, y = fix_pair_person(5, 5)
        assert x != y
        assert 1 <= x <= PERSON_NUMBER
        assert 1 <= y <= PERSON_NUMBER

    def test_fix_pair_person_already_different(self):
        """Test that already different IDs are unchanged."""
        x, y = fix_pair_person(1, 2)
        assert x == 1
        assert y == 2

    def test_fix_pair_sign_different(self):
        """Test that fix_pair_sign returns different IDs."""
        x, y = fix_pair_sign(3, 3)
        assert x != y
        assert 1 <= x <= SIGN_NUMBER_EACH
        assert 1 <= y <= SIGN_NUMBER_EACH

    def test_fix_pair_sign_already_different(self):
        """Test that already different signature IDs are unchanged."""
        x, y = fix_pair_sign(1, 5)
        assert x == 1
        assert y == 5


class TestImageProcessing:
    """Test suite for image processing functions."""

    def test_convert_to_image_tensor_shape(self):
        """Test tensor conversion output shape."""
        image_array = np.random.randint(0, 256, (155, 220), dtype=np.uint8)
        tensor = convert_to_image_tensor(image_array)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 220, 155)

    def test_convert_to_image_tensor_normalization(self):
        """Test that tensor values are normalized to [0, 1]."""
        image_array = np.full((155, 220), 255, dtype=np.uint8)
        tensor = convert_to_image_tensor(image_array)

        assert tensor.max() <= 1.0
        assert tensor.min() >= 0.0
        assert torch.allclose(tensor, torch.ones_like(tensor), atol=0.01)

    def test_convert_to_image_tensor_zeros(self):
        """Test conversion of zero array."""
        image_array = np.zeros((155, 220), dtype=np.uint8)
        tensor = convert_to_image_tensor(image_array)

        assert torch.allclose(tensor, torch.zeros_like(tensor))

    def test_invert_image_shape(self):
        """Test invert_image output shape."""
        # Create dummy image
        img = Image.new("RGB", (300, 200), color="white")
        inverted = invert_image(img)

        assert inverted.shape == (155, 220)
        assert inverted.dtype == np.uint8

    def test_invert_image_binary(self):
        """Test that inverted image is binary (0 or 255)."""
        img = Image.new("L", (300, 200), color=128)
        inverted = invert_image(img)

        unique_values = np.unique(inverted)
        assert all(val in [0, 255] for val in unique_values)

    def test_invert_image_threshold(self):
        """Test threshold behavior in image inversion."""
        # Create image with values below threshold
        img = Image.new("L", (300, 200), color=30)
        inverted = invert_image(img)

        # After inversion, low values should become high, then thresholded
        assert np.all((inverted == 0) | (inverted == 255))


class TestConstants:
    """Test suite for module constants."""

    def test_person_number_valid(self):
        """Test PERSON_NUMBER is a positive integer."""
        assert isinstance(PERSON_NUMBER, int)
        assert PERSON_NUMBER > 0

    def test_sign_number_each_valid(self):
        """Test SIGN_NUMBER_EACH is a positive integer."""
        assert isinstance(SIGN_NUMBER_EACH, int)
        assert SIGN_NUMBER_EACH > 0


class TestImageProcessingEdgeCases:
    """Test edge cases in image processing."""

    @pytest.mark.parametrize(
        "color",
        [0, 50, 100, 150, 200, 255],
    )
    def test_invert_image_various_colors(self, color):
        """Test image inversion with various grayscale values."""
        img = Image.new("L", (300, 200), color=color)
        inverted = invert_image(img)

        assert inverted.shape == (155, 220)
        assert inverted.dtype == np.uint8

    def test_convert_tensor_preserves_dtype(self):
        """Test that tensor conversion maintains float32 precision."""
        image_array = np.random.randint(0, 256, (155, 220), dtype=np.uint8)
        tensor = convert_to_image_tensor(image_array)

        assert tensor.dtype == torch.float32
